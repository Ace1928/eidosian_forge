import code
import sys
import tokenize
from io import BytesIO
from traceback import format_exception
from types import TracebackType
from typing import Type
from twisted.conch import recvline
from twisted.internet import defer
from twisted.python.compat import _get_async_param
from twisted.python.htmlizer import TokenPrinter
from twisted.python.monkey import MonkeyPatcher
class ManholeInterpreter(code.InteractiveInterpreter):
    """
    Interactive Interpreter with special output and Deferred support.

    Aside from the features provided by L{code.InteractiveInterpreter}, this
    class captures sys.stdout output and redirects it to the appropriate
    location (the Manhole protocol instance).  It also treats Deferreds
    which reach the top-level specially: each is formatted to the user with
    a unique identifier and a new callback and errback added to it, each of
    which will format the unique identifier and the result with which the
    Deferred fires and then pass it on to the next participant in the
    callback chain.
    """
    numDeferreds = 0

    def __init__(self, handler, locals=None, filename='<console>'):
        code.InteractiveInterpreter.__init__(self, locals)
        self._pendingDeferreds = {}
        self.handler = handler
        self.filename = filename
        self.resetBuffer()
        self.monkeyPatcher = MonkeyPatcher()
        self.monkeyPatcher.addPatch(sys, 'displayhook', self.displayhook)
        self.monkeyPatcher.addPatch(sys, 'excepthook', self.excepthook)
        self.monkeyPatcher.addPatch(sys, 'stdout', FileWrapper(self.handler))

    def resetBuffer(self):
        """
        Reset the input buffer.
        """
        self.buffer = []

    def push(self, line):
        """
        Push a line to the interpreter.

        The line should not have a trailing newline; it may have
        internal newlines.  The line is appended to a buffer and the
        interpreter's runsource() method is called with the
        concatenated contents of the buffer as source.  If this
        indicates that the command was executed or invalid, the buffer
        is reset; otherwise, the command is incomplete, and the buffer
        is left as it was after the line was appended.  The return
        value is 1 if more input is required, 0 if the line was dealt
        with in some way (this is the same as runsource()).

        @param line: line of text
        @type line: L{bytes}
        @return: L{bool} from L{code.InteractiveInterpreter.runsource}
        """
        self.buffer.append(line)
        source = b'\n'.join(self.buffer)
        source = source.decode('utf-8')
        more = self.runsource(source, self.filename)
        if not more:
            self.resetBuffer()
        return more

    def runcode(self, *a, **kw):
        with self.monkeyPatcher:
            code.InteractiveInterpreter.runcode(self, *a, **kw)

    def excepthook(self, excType: Type[BaseException], excValue: BaseException, excTraceback: TracebackType) -> None:
        """
        Format exception tracebacks and write them to the output handler.
        """
        lines = format_exception(excType, excValue, excTraceback.tb_next)
        self.write(''.join(lines))

    def displayhook(self, obj):
        self.locals['_'] = obj
        if isinstance(obj, defer.Deferred):
            if hasattr(obj, 'result'):
                self.write(repr(obj))
            elif id(obj) in self._pendingDeferreds:
                self.write('<Deferred #%d>' % (self._pendingDeferreds[id(obj)][0],))
            else:
                d = self._pendingDeferreds
                k = self.numDeferreds
                d[id(obj)] = (k, obj)
                self.numDeferreds += 1
                obj.addCallbacks(self._cbDisplayDeferred, self._ebDisplayDeferred, callbackArgs=(k, obj), errbackArgs=(k, obj))
                self.write('<Deferred #%d>' % (k,))
        elif obj is not None:
            self.write(repr(obj))

    def _cbDisplayDeferred(self, result, k, obj):
        self.write('Deferred #%d called back: %r' % (k, result), True)
        del self._pendingDeferreds[id(obj)]
        return result

    def _ebDisplayDeferred(self, failure, k, obj):
        self.write('Deferred #%d failed: %r' % (k, failure.getErrorMessage()), True)
        del self._pendingDeferreds[id(obj)]
        return failure

    def write(self, data, isAsync=None, **kwargs):
        isAsync = _get_async_param(isAsync, **kwargs)
        self.handler.addOutput(data, isAsync)
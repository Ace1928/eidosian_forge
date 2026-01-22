import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
class ManholeLoopbackMixin:
    serverProtocol = manhole.ColoredManhole

    def test_SimpleExpression(self):
        """
        Evaluate simple expression.
        """
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b'1 + 1\ndone')

        def finished(ign):
            self._assertBuffer([b'>>> 1 + 1', b'2', b'>>> done'])
        return done.addCallback(finished)

    def test_TripleQuoteLineContinuation(self):
        """
        Evaluate line continuation in triple quotes.
        """
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b"'''\n'''\ndone")

        def finished(ign):
            self._assertBuffer([b">>> '''", b"... '''", b"'\\n'", b'>>> done'])
        return done.addCallback(finished)

    def test_FunctionDefinition(self):
        """
        Evaluate function definition.
        """
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b'def foo(bar):\n\tprint(bar)\n\nfoo(42)\ndone')

        def finished(ign):
            self._assertBuffer([b'>>> def foo(bar):', b'...     print(bar)', b'... ', b'>>> foo(42)', b'42', b'>>> done'])
        return done.addCallback(finished)

    def test_ClassDefinition(self):
        """
        Evaluate class definition.
        """
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b"class Foo:\n\tdef bar(self):\n\t\tprint('Hello, world!')\n\nFoo().bar()\ndone")

        def finished(ign):
            self._assertBuffer([b'>>> class Foo:', b'...     def bar(self):', b"...         print('Hello, world!')", b'... ', b'>>> Foo().bar()', b'Hello, world!', b'>>> done'])
        return done.addCallback(finished)

    def test_Exception(self):
        """
        Evaluate raising an exception.
        """
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b"raise Exception('foo bar baz')\ndone")

        def finished(ign):
            self._assertBuffer([b">>> raise Exception('foo bar baz')", b'Traceback (most recent call last):', b'  File "<console>", line 1, in ' + defaultFunctionName.encode('utf-8'), b'Exception: foo bar baz', b'>>> done'])
        done.addCallback(finished)
        return done

    def test_ExceptionWithCustomExcepthook(self):
        """
        Raised exceptions are handled the same way even if L{sys.excepthook}
        has been modified from its original value.
        """
        self.patch(sys, 'excepthook', lambda *args: None)
        return self.test_Exception()

    def test_ControlC(self):
        """
        Evaluate interrupting with CTRL-C.
        """
        done = self.recvlineClient.expect(b'done')
        self._testwrite(b'cancelled line' + manhole.CTRL_C + b'done')

        def finished(ign):
            self._assertBuffer([b'>>> cancelled line', b'KeyboardInterrupt', b'>>> done'])
        return done.addCallback(finished)

    def test_interruptDuringContinuation(self):
        """
        Sending ^C to Manhole while in a state where more input is required to
        complete a statement should discard the entire ongoing statement and
        reset the input prompt to the non-continuation prompt.
        """
        continuing = self.recvlineClient.expect(b'things')
        self._testwrite(b'(\nthings')

        def gotContinuation(ignored):
            self._assertBuffer([b'>>> (', b'... things'])
            interrupted = self.recvlineClient.expect(b'>>> ')
            self._testwrite(manhole.CTRL_C)
            return interrupted
        continuing.addCallback(gotContinuation)

        def gotInterruption(ignored):
            self._assertBuffer([b'>>> (', b'... things', b'KeyboardInterrupt', b'>>> '])
        continuing.addCallback(gotInterruption)
        return continuing

    def test_ControlBackslash(self):
        """
        Evaluate cancelling with CTRL-\\.
        """
        self._testwrite(b'cancelled line')
        partialLine = self.recvlineClient.expect(b'cancelled line')

        def gotPartialLine(ign):
            self._assertBuffer([b'>>> cancelled line'])
            self._testwrite(manhole.CTRL_BACKSLASH)
            d = self.recvlineClient.onDisconnection
            return self.assertFailure(d, error.ConnectionDone)

        def gotClearedLine(ign):
            self._assertBuffer([b''])
        return partialLine.addCallback(gotPartialLine).addCallback(gotClearedLine)

    @defer.inlineCallbacks
    def test_controlD(self):
        """
        A CTRL+D in the middle of a line doesn't close a connection,
        but at the beginning of a line it does.
        """
        self._testwrite(b'1 + 1')
        yield self.recvlineClient.expect(b'\\+ 1')
        self._assertBuffer([b'>>> 1 + 1'])
        self._testwrite(manhole.CTRL_D + b' + 1')
        yield self.recvlineClient.expect(b'\\+ 1')
        self._assertBuffer([b'>>> 1 + 1 + 1'])
        self._testwrite(b'\n')
        yield self.recvlineClient.expect(b'3\n>>> ')
        self._testwrite(manhole.CTRL_D)
        d = self.recvlineClient.onDisconnection
        yield self.assertFailure(d, error.ConnectionDone)

    @defer.inlineCallbacks
    def test_ControlL(self):
        """
        CTRL+L is generally used as a redraw-screen command in terminal
        applications.  Manhole doesn't currently respect this usage of it,
        but it should at least do something reasonable in response to this
        event (rather than, say, eating your face).
        """
        self._testwrite(b'\n1 + 1')
        yield self.recvlineClient.expect(b'\\+ 1')
        self._assertBuffer([b'>>> ', b'>>> 1 + 1'])
        self._testwrite(manhole.CTRL_L + b' + 1')
        yield self.recvlineClient.expect(b'1 \\+ 1 \\+ 1')
        self._assertBuffer([b'>>> 1 + 1 + 1'])

    def test_controlA(self):
        """
        CTRL-A can be used as HOME - returning cursor to beginning of
        current line buffer.
        """
        self._testwrite(b'rint "hello"' + b'\x01' + b'p')
        d = self.recvlineClient.expect(b'print "hello"')

        def cb(ignore):
            self._assertBuffer([b'>>> print "hello"'])
        return d.addCallback(cb)

    def test_controlE(self):
        """
        CTRL-E can be used as END - setting cursor to end of current
        line buffer.
        """
        self._testwrite(b'rint "hello' + b'\x01' + b'p' + b'\x05' + b'"')
        d = self.recvlineClient.expect(b'print "hello"')

        def cb(ignore):
            self._assertBuffer([b'>>> print "hello"'])
        return d.addCallback(cb)

    @defer.inlineCallbacks
    def test_deferred(self):
        """
        When a deferred is returned to the manhole REPL, it is displayed with
        a sequence number, and when the deferred fires, the result is printed.
        """
        self._testwrite(b'from twisted.internet import defer, reactor\nd = defer.Deferred()\nd\n')
        yield self.recvlineClient.expect(b'<Deferred #0>')
        self._testwrite(b"c = reactor.callLater(0.1, d.callback, 'Hi!')\n")
        yield self.recvlineClient.expect(b'>>> ')
        yield self.recvlineClient.expect(b"Deferred #0 called back: 'Hi!'\n>>> ")
        self._assertBuffer([b'>>> from twisted.internet import defer, reactor', b'>>> d = defer.Deferred()', b'>>> d', b'<Deferred #0>', b">>> c = reactor.callLater(0.1, d.callback, 'Hi!')", b"Deferred #0 called back: 'Hi!'", b'>>> '])
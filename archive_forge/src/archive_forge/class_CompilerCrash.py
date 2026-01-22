from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
class CompilerCrash(CompileError):

    def __init__(self, pos, context, message, cause, stacktrace=None):
        if message:
            message = u'\n' + message
        else:
            message = u'\n'
        self.message_only = message
        if context:
            message = u'Compiler crash in %s%s' % (context, message)
        if stacktrace:
            import traceback
            message += u'\n\nCompiler crash traceback from this point on:\n' + u''.join(traceback.format_tb(stacktrace))
        if cause:
            if not stacktrace:
                message += u'\n'
            message += u'%s: %s' % (cause.__class__.__name__, cause)
        CompileError.__init__(self, pos, message)
        self.args = (pos, context, message, cause, stacktrace)
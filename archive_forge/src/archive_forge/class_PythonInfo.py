from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class PythonInfo(LanguageInfo):

    def pyframe(self, frame):
        pyframe = Frame(frame).get_pyop()
        if pyframe:
            return pyframe
        else:
            raise gdb.RuntimeError('Unable to find the Python frame, run your code with a debug build (configure with --with-pydebug or compile with -g).')

    def lineno(self, frame):
        return self.pyframe(frame).current_line_num()

    def is_relevant_function(self, frame):
        return Frame(frame).is_evalframeex()

    def get_source_line(self, frame):
        try:
            pyframe = self.pyframe(frame)
            return '%4d    %s' % (pyframe.current_line_num(), pyframe.current_line().rstrip())
        except IOError:
            return None

    def exc_info(self, frame):
        try:
            tstate = frame.read_var('tstate').dereference()
            if gdb.parse_and_eval('tstate->frame == f'):
                if sys.version_info >= (3, 12, 0, 'alpha', 6):
                    inf_type = inf_value = tstate['current_exception']
                else:
                    inf_type = tstate['curexc_type']
                    inf_value = tstate['curexc_value']
                if inf_type:
                    return 'An exception was raised: %s' % (inf_value,)
        except (ValueError, RuntimeError):
            pass

    def static_break_functions(self):
        yield 'PyEval_EvalFrameEx'
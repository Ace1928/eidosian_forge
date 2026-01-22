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
class PyBreak(gdb.Command):
    """
    Set a Python breakpoint. Examples:

    Break on any function or method named 'func' in module 'modname'

        py-break modname.func

    Break on any function or method named 'func'

        py-break func
    """

    @dont_suppress_errors
    def invoke(self, funcname, from_tty):
        if '.' in funcname:
            modname, dot, funcname = funcname.rpartition('.')
            cond = '$pyname_equals("%s") && $pymod_equals("%s")' % (funcname, modname)
        else:
            cond = '$pyname_equals("%s")' % funcname
        gdb.execute('break PyEval_EvalFrameEx if ' + cond)
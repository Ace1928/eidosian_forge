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
class PyGlobals(gdb.Command):
    """List all the globals in the currently select Python frame"""

    def __init__(self):
        gdb.Command.__init__(self, 'py-globals', gdb.COMMAND_DATA, gdb.COMPLETE_NONE)

    @dont_suppress_errors
    def invoke(self, args, from_tty):
        name = str(args)
        frame = Frame.get_selected_python_frame()
        if not frame:
            print('Unable to locate python frame')
            return
        pyop_frame = frame.get_pyop()
        if not pyop_frame:
            print(UNABLE_READ_INFO_PYTHON_FRAME)
            return
        for pyop_name, pyop_value in pyop_frame.iter_locals():
            print('%s = %s' % (pyop_name.proxyval(set()), pyop_value.get_truncated_repr(MAX_OUTPUT_LEN)))

    def get_namespace(self, pyop_frame):
        return pyop_frame.iter_globals()
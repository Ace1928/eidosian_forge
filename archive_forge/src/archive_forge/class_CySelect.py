from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CySelect(CythonCommand):
    """
    Select a frame. Use frame numbers as listed in `cy backtrace`.
    This command is useful because `cy backtrace` prints a reversed backtrace.
    """
    name = 'cy select'

    @libpython.dont_suppress_errors
    def invoke(self, stackno, from_tty):
        try:
            stackno = int(stackno)
        except ValueError:
            raise gdb.GdbError('Not a valid number: %r' % (stackno,))
        frame = gdb.selected_frame()
        while frame.newer():
            frame = frame.newer()
        stackdepth = libpython.stackdepth(frame)
        try:
            gdb.execute('select %d' % (stackdepth - stackno - 1,))
        except RuntimeError as e:
            raise gdb.GdbError(*e.args)
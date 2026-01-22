from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyCValue(CyCName):
    """
    Get the value of a Cython variable.
    """

    @libpython.dont_suppress_errors
    @require_cython_frame
    @gdb_function_value_to_unicode
    def invoke(self, cyname, frame=None):
        globals_dict = self.get_cython_globals_dict()
        cython_function = self.get_cython_function(frame)
        if self.is_initialized(cython_function, cyname):
            cname = super(CyCValue, self).invoke(cyname, frame=frame)
            return gdb.parse_and_eval(cname)
        elif cyname in globals_dict:
            return globals_dict[cyname]._gdbval
        else:
            raise gdb.GdbError('Variable %s is not initialized.' % cyname)
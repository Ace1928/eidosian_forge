from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def _evalcode_cython(self, executor, code, input_type):
    with libpython.FetchAndRestoreError():
        global_dict = gdb.parse_and_eval('(PyObject *) PyModule_GetDict(__pyx_m)')
        local_dict = gdb.parse_and_eval('(PyObject *) PyDict_New()')
        try:
            self._fill_locals_dict(executor, libpython.pointervalue(local_dict))
            result = executor.evalcode(code, input_type, global_dict, local_dict)
        finally:
            executor.xdecref(libpython.pointervalue(local_dict))
    return result
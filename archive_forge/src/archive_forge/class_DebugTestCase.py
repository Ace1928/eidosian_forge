from __future__ import absolute_import
import os
import re
import sys
import trace
import inspect
import warnings
import unittest
import textwrap
import tempfile
import functools
import traceback
import itertools
import gdb
from .. import libcython
from .. import libpython
from . import TestLibCython as test_libcython
from ...Utils import add_metaclass
@add_metaclass(TraceMethodCallMeta)
class DebugTestCase(unittest.TestCase):
    """
    Base class for test cases. On teardown it kills the inferior and unsets
    all breakpoints.
    """

    def __init__(self, name):
        super(DebugTestCase, self).__init__(name)
        self.cy = libcython.cy
        self.module = libcython.cy.cython_namespace['codefile']
        self.spam_func, self.spam_meth = libcython.cy.functions_by_name['spam']
        self.ham_func = libcython.cy.functions_by_qualified_name['codefile.ham']
        self.eggs_func = libcython.cy.functions_by_qualified_name['codefile.eggs']

    def read_var(self, varname, cast_to=None):
        result = gdb.parse_and_eval('$cy_cvalue("%s")' % varname)
        if cast_to:
            result = cast_to(result)
        return result

    def local_info(self):
        return gdb.execute('info locals', to_string=True)

    def lineno_equals(self, source_line=None, lineno=None):
        if source_line is not None:
            lineno = test_libcython.source_to_lineno[source_line]
        frame = gdb.selected_frame()
        self.assertEqual(libcython.cython_info.lineno(frame), lineno)

    def break_and_run(self, source_line):
        break_lineno = test_libcython.source_to_lineno[source_line]
        gdb.execute('cy break codefile:%d' % break_lineno, to_string=True)
        gdb.execute('run', to_string=True)

    def tearDown(self):
        gdb.execute('delete breakpoints', to_string=True)
        try:
            gdb.execute('kill inferior 1', to_string=True)
        except RuntimeError:
            pass
        gdb.execute('set args -c "import codefile"')
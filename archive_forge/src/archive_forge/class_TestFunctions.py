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
class TestFunctions(DebugTestCase):

    def test_functions(self):
        self.break_and_run('c = 2')
        result = gdb.execute('print $cy_cname("b")', to_string=True)
        assert re.search('__pyx_.*b', result), result
        result = gdb.execute('print $cy_lineno()', to_string=True)
        supposed_lineno = test_libcython.source_to_lineno['c = 2']
        assert str(supposed_lineno) in result, (supposed_lineno, result)
        result = gdb.execute('print $cy_cvalue("b")', to_string=True)
        assert '= 1' in result
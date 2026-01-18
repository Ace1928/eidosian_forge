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
def test_python_step(self):
    self.break_and_run('os.path.join("foo", "bar")')
    result = gdb.execute('cy step', to_string=True)
    curframe = gdb.selected_frame()
    self.assertEqual(curframe.name(), 'PyEval_EvalFrameEx')
    pyframe = libpython.Frame(curframe).get_pyop()
    frame_name = pyframe.co_name.proxyval(set())
    self.assertEqual(frame_name, 'join')
    assert re.match('\\d+    def join\\(', result), result
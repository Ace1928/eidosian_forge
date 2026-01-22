from test import support
from test.support import import_helper
import_helper.import_module('_multiprocessing')
import importlib
import importlib.machinery
import unittest
import sys
import os
import os.path
import py_compile
from test.support import os_helper
from test.support.script_helper import (
import multiprocess as multiprocessing
import_helper.import_module('multiprocess.synchronize')
import sys
import time
from multiprocess import Pool, set_start_method
import sys
import time
from multiprocess import Pool, set_start_method
import sys, os.path, runpy
def _check_output(self, script_name, exit_code, out, err):
    if verbose > 1:
        print('Output from test script %r:' % script_name)
        print(repr(out))
    self.assertEqual(exit_code, 0)
    self.assertEqual(err.decode('utf-8'), '')
    expected_results = '%s -> [1, 4, 9]' % self.start_method
    self.assertEqual(out.decode('utf-8').strip(), expected_results)
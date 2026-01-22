from __future__ import print_function
from __future__ import absolute_import
import subprocess
import unittest
import greenlet
from . import _test_extension_cpp
from . import TestCase
from . import WIN
def _do_test_unhandled_exception(self, target):
    import os
    import sys
    script = os.path.join(os.path.dirname(__file__), 'fail_cpp_exception.py')
    args = [sys.executable, script, target.__name__ if not isinstance(target, str) else target]
    __traceback_info__ = args
    with self.assertRaises(subprocess.CalledProcessError) as exc:
        subprocess.check_output(args, encoding='utf-8', stderr=subprocess.STDOUT)
    ex = exc.exception
    expected_exit = self.get_expected_returncodes_for_aborted_process()
    self.assertIn(ex.returncode, expected_exit)
    self.assertIn('fail_cpp_exception is running', ex.output)
    return ex.output
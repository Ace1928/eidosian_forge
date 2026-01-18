from __future__ import print_function
from __future__ import absolute_import
import subprocess
import unittest
import greenlet
from . import _test_extension_cpp
from . import TestCase
from . import WIN
@unittest.skipIf(WIN, 'XXX: This does not crash on Windows')
def test_unhandled_std_exception_as_greenlet_function_aborts(self):
    output = self._do_test_unhandled_exception('run_as_greenlet_target')
    self.assertIn('Thrown from an extension.', output)
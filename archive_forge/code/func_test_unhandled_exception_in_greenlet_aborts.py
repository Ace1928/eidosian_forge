from __future__ import print_function
from __future__ import absolute_import
import subprocess
import unittest
import greenlet
from . import _test_extension_cpp
from . import TestCase
from . import WIN
def test_unhandled_exception_in_greenlet_aborts(self):
    self._do_test_unhandled_exception('run_unhandled_exception_in_greenlet_aborts')
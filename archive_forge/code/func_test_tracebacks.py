import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_tracebacks(self):

    @patch.object(Foo, 'f', object())
    def test():
        raise AssertionError
    try:
        test()
    except:
        err = sys.exc_info()
    result = unittest.TextTestResult(None, None, 0)
    traceback = result._exc_info_to_string(err, self)
    self.assertIn('raise AssertionError', traceback)
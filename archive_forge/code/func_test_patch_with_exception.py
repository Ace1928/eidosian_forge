import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_with_exception(self):
    foo = {}

    @patch.dict(foo, {'a': 'b'})
    def test():
        raise NameError('Konrad')
    try:
        test()
    except NameError:
        pass
    else:
        self.fail('NameError not raised by test')
    self.assertEqual(foo, {})
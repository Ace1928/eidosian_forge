import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_nested_autospec_repr(self):
    p = patch('mock.tests.support', autospec=True)
    m = p.start()
    try:
        self.assertIn(" name='support.SomeClass.wibble()'", repr(m.SomeClass.wibble()))
        self.assertIn(" name='support.SomeClass().wibble()'", repr(m.SomeClass().wibble()))
    finally:
        p.stop()
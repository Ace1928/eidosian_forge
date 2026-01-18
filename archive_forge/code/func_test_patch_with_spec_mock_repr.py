import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_with_spec_mock_repr(self):
    for arg in ('spec', 'autospec', 'spec_set'):
        p = patch('%s.SomeClass' % __name__, **{arg: True})
        m = p.start()
        try:
            self.assertIn(" name='SomeClass'", repr(m))
            self.assertIn(" name='SomeClass.class_attribute'", repr(m.class_attribute))
            self.assertIn(" name='SomeClass()'", repr(m()))
            self.assertIn(" name='SomeClass().class_attribute'", repr(m().class_attribute))
        finally:
            p.stop()
import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
def test_patch_spec_set_instance(self):
    patcher = patch('%s.X' % __name__, spec_set=X())
    mock = patcher.start()
    self.addCleanup(patcher.stop)
    self.assertNotCallable(mock)
    self.assertRaises(TypeError, mock)
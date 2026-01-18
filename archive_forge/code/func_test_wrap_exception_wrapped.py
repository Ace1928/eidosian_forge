from unittest import mock
from oslo_versionedobjects import exception
from oslo_versionedobjects import test
def test_wrap_exception_wrapped(self):
    test = TestWrapper()
    self.assertTrue(hasattr(test.raise_exc, '__wrapped__'))
import unittest
from unittest import mock
def test_sealed_exception_has_attribute_name(self):
    m = mock.Mock()
    mock.seal(m)
    with self.assertRaises(AttributeError) as cm:
        m.SECRETE_name
    self.assertIn('SECRETE_name', str(cm.exception))
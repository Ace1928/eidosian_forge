import unittest
from unittest import mock
def test_seals_recurse_on_added_attributes(self):
    m = mock.Mock()
    m.test1.test2().test3 = 4
    mock.seal(m)
    self.assertEqual(m.test1.test2().test3, 4)
    with self.assertRaises(AttributeError):
        m.test1.test2().test4
    with self.assertRaises(AttributeError):
        m.test1.test3
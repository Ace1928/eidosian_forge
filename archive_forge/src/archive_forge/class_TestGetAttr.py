import unittest
from traits.api import HasTraits, Int
class TestGetAttr(unittest.TestCase):

    def setUp(self):
        self.a = A()

    def test_bad__getattribute__(self):
        self.assertEqual(self.a.__getattribute__('x'), 5)
        with self.assertRaises(TypeError) as e:
            self.a.__getattribute__(2)
        exception_msg = str(e.exception)
        self.assertIn('2', exception_msg)
        self.assertIn('int', exception_msg)
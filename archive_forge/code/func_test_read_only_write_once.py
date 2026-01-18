import unittest
from traits.api import HasTraits, Str, Undefined, ReadOnly, Float
def test_read_only_write_once(self):
    f = Foo()
    self.assertEqual(f.name, '')
    self.assertIs(f.original_name, Undefined)
    f.name = 'first'
    self.assertEqual(f.name, 'first')
    self.assertEqual(f.original_name, 'first')
    f.name = 'second'
    self.assertEqual(f.name, 'second')
    self.assertEqual(f.original_name, 'first')
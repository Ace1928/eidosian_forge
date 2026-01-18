import unittest
from traits.api import HasTraits
def test_read_write_property(self):
    model = Model()
    self.assertEqual(model.value, 0)
    model.value = 23
    self.assertEqual(model.value, 23)
    model.value = 77
    self.assertEqual(model.value, 77)
    with self.assertRaises(AttributeError):
        del model.value
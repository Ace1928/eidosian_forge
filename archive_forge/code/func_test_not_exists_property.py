import unittest
def test_not_exists_property(self):
    from kivy.uix.behaviors.knspace import knspace
    self.assertRaises(AttributeError, lambda: knspace.label2)
    knspace.property('label2')
    self.assertIsNone(knspace.label2)
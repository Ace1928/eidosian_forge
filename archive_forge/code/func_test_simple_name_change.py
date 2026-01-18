import unittest
def test_simple_name_change(self):
    from kivy.lang import Builder
    from kivy.uix.behaviors.knspace import knspace
    w = Builder.load_string("\n<NamedLabel@KNSpaceBehavior+Label>\n\nNamedLabel:\n    knsname: 'label8'\n    text: 'Hello'\n")
    self.assertEqual(w, knspace.label8)
    w.knsname = 'named_label8'
    self.assertIsNone(knspace.label8)
    self.assertEqual(w, knspace.named_label8)
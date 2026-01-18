import unittest
from kivy.uix.stacklayout import StackLayout
from kivy.uix.widget import Widget
def test_stacklayout_fixed_size(self):
    sl = StackLayout()
    wgts = [Widget(size=(50, 50), size_hint=(None, None)) for i in range(4)]
    for wgt in wgts:
        sl.add_widget(wgt)
    sl.do_layout()
    self.assertEqual(wgts[0].pos, [0, sl.height / 2.0])
    self.assertEqual(wgts[1].pos, [sl.width / 2.0, sl.height / 2.0])
    self.assertEqual(wgts[2].pos, [0, 0])
    self.assertEqual(wgts[3].pos, [sl.width / 2.0, 0])
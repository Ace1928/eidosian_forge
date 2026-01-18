import unittest
from kivy.uix.stacklayout import StackLayout
from kivy.uix.widget import Widget
def test_stacklayout_overflow(self):
    sl = StackLayout()
    wgts = [Widget(size_hint=(0.2 * i, 0.2 * i)) for i in range(1, 4)]
    for wgt in wgts:
        sl.add_widget(wgt)
    sl.padding = 5
    sl.spacing = 5
    sl.do_layout()
    self.assertEqual(wgts[0].pos, [5, 77])
    self.assertEqual(wgts[1].pos, [27, 59])
    self.assertAlmostEqual(wgts[2].pos[0], 5)
    self.assertAlmostEqual(wgts[2].pos[1], 0)
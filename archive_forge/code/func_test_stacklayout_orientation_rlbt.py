import unittest
from kivy.uix.stacklayout import StackLayout
from kivy.uix.widget import Widget
def test_stacklayout_orientation_rlbt(self):
    sl = StackLayout()
    wgts = [Widget(size_hint=(0.5, 0.5)) for i in range(4)]
    for wgt in wgts:
        sl.add_widget(wgt)
    sl.orientation = 'rl-bt'
    sl.do_layout()
    self.assertEqual(wgts[0].pos, [sl.width / 2.0, 0])
    self.assertEqual(wgts[1].pos, [0, 0])
    self.assertEqual(wgts[2].pos, [sl.width / 2.0, sl.height / 2.0])
    self.assertEqual(wgts[3].pos, [0, sl.height / 2.0])
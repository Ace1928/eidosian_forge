import unittest
from kivy.base import EventLoop
from kivy.tests import UTMotionEvent
from kivy.uix.relativelayout import RelativeLayout
def test_relativelayout_coordinates(self):
    EventLoop.ensure_window()
    rl = RelativeLayout(pos=(100, 100))
    EventLoop.window.add_widget(rl)
    self.assertEqual(rl.to_parent(50, 50), (150, 150))
    self.assertEqual(rl.to_local(50, 50), (-50, -50))
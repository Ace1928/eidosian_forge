import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_scroll_doesnt_move_cursor(self):
    ti = self.make_scrollable_text_input()
    from kivy.base import EventLoop
    win = EventLoop.window
    touch = UTMotionEvent('unittest', next(touch_id), {'x': ti.center_x / float(win.width), 'y': ti.center_y / float(win.height)})
    touch.profile.append('button')
    touch.button = 'scrolldown'
    prev_cursor = ti.cursor
    assert ti._visible_lines_range == (20, 30)
    EventLoop.post_dispatch_input('begin', touch)
    EventLoop.post_dispatch_input('end', touch)
    self.advance_frames(1)
    assert ti._visible_lines_range == (20 - ti.lines_to_scroll, 30 - ti.lines_to_scroll)
    assert ti.cursor == prev_cursor
import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_vertical_scroll_doesnt_depend_on_lines_rendering(self):
    ti = self.make_scrollable_text_input()
    ti.do_cursor_movement('cursor_home', control=True)
    self.advance_frames(1)
    assert ti._visible_lines_range == (0, 10)
    from kivy.base import EventLoop
    win = EventLoop.window
    for _ in range(0, 30, ti.lines_to_scroll):
        touch = UTMotionEvent('unittest', next(touch_id), {'x': ti.center_x / float(win.width), 'y': ti.center_y / float(win.height)})
        touch.profile.append('button')
        touch.button = 'scrollup'
        EventLoop.post_dispatch_input('begin', touch)
        EventLoop.post_dispatch_input('end', touch)
        self.advance_frames(1)
    assert ti._visible_lines_range == (20, 30)
    ti.do_cursor_movement('cursor_home', control=True)
    ti._trigger_update_graphics()
    self.advance_frames(1)
    assert ti._visible_lines_range == (0, 10)
    touch = UTMotionEvent('unittest', next(touch_id), {'x': ti.center_x / float(win.width), 'y': ti.center_y / float(win.height)})
    touch.profile.append('button')
    touch.button = 'scrollup'
    EventLoop.post_dispatch_input('begin', touch)
    EventLoop.post_dispatch_input('end', touch)
    self.advance_frames(1)
    assert ti._visible_lines_range == (ti.lines_to_scroll, 10 + ti.lines_to_scroll)
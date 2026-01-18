import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_selectall_copy_paste(self):
    text = 'test'
    ti = TextInput(multiline=False, text=text)
    ti.focus = True
    self.render(ti)
    from kivy.base import EventLoop
    win = EventLoop.window
    win.dispatch('on_key_down', 97, 4, 'a', ['capslock', 'ctrl'])
    win.dispatch('on_key_up', 97, 4)
    self.advance_frames(1)
    win.dispatch('on_key_down', 99, 6, 'c', ['capslock', 'numlock', 'ctrl'])
    win.dispatch('on_key_up', 99, 6)
    self.advance_frames(1)
    win.dispatch('on_key_down', 278, 74, None, ['capslock'])
    win.dispatch('on_key_up', 278, 74)
    self.advance_frames(1)
    win.dispatch('on_key_down', 118, 25, 'v', ['numlock', 'ctrl'])
    win.dispatch('on_key_up', 118, 25)
    self.advance_frames(1)
    assert ti.text == 'testtest'
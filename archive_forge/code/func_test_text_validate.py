import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_text_validate(self):
    ti = TextInput(multiline=False)
    ti.focus = True
    self.render(ti)
    self.assertFalse(ti.multiline)
    self.assertTrue(ti.focus)
    self.assertTrue(ti.text_validate_unfocus)
    ti.validate_test = None
    ti.bind(on_text_validate=lambda *_: setattr(ti, 'validate_test', True))
    ti._key_down((None, None, 'enter', 1), repeat=False)
    self.assertTrue(ti.validate_test)
    self.assertFalse(ti.focus)
    ti.validate_test = None
    ti.text_validate_unfocus = False
    ti.focus = True
    self.assertTrue(ti.focus)
    ti._key_down((None, None, 'enter', 1), repeat=False)
    self.assertTrue(ti.validate_test)
    self.assertTrue(ti.focus)
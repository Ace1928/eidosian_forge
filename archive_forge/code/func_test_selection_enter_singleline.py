import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_selection_enter_singleline(self):
    text = 'singleline'
    ti = TextInput(multiline=False, text=text)
    ti.focus = True
    self.render(ti)
    self.assertTrue(ti.focus)
    self.assertEqual(ti.cursor, (len(text), 0))
    steps = 4
    options = (('enter', text), ('backspace', text[:len(text) - steps]))
    for key, txt in options:
        ti._key_down((None, None, 'shift', 1), repeat=False)
        for _ in range(steps):
            ti._key_down((None, None, 'cursor_left', 1), repeat=False)
        ti._key_up((None, None, 'shift', 1), repeat=False)
        self.assertEqual(ti.cursor, (len(text[:-steps]), 0))
        self.assertEqual(ti.text, text)
        ti._key_down((None, None, key, 1), repeat=False)
        self.assertEqual(ti.text, txt)
        ti._key_down((None, None, 'cursor_end', 1), repeat=False)
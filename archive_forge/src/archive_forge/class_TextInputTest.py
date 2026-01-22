import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
class TextInputTest(unittest.TestCase):

    def test_focusable_when_disabled(self):
        ti = TextInput()
        ti.disabled = True
        ti.focused = True
        ti.bind(focus=self.on_focused)

    def on_focused(self, instance, value):
        self.assertTrue(instance.focused, value)

    def test_wordbreak(self):
        self.test_txt = 'Firstlongline\n\nSecondveryverylongline'
        ti = TextInput(width='30dp', size_hint_x=None)
        ti.bind(text=self.on_text)
        ti.text = self.test_txt

    def on_text(self, instance, value):
        self.assertEqual(instance.text, self.test_txt)
        pos_S = self.test_txt.index('S')
        self.assertEqual(instance.get_cursor_from_index(pos_S), (0, 6))
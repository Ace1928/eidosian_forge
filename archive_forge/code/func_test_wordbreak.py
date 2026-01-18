import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_wordbreak(self):
    self.test_txt = 'Firstlongline\n\nSecondveryverylongline'
    ti = TextInput(width='30dp', size_hint_x=None)
    ti.bind(text=self.on_text)
    ti.text = self.test_txt
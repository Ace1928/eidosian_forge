import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def ti_height_for_x_lines(ti, x):
    """Calculate TextInput height required to display x lines in viewport.

    ti -- TextInput object being used
    x -- number of lines to display
    """
    padding_top = ti.padding[1]
    padding_bottom = ti.padding[3]
    return int((ti.line_height + ti.line_spacing) * x + padding_top + padding_bottom)
from math import cos, sin, pi, sqrt, atan
from colorsys import rgb_to_hsv, hsv_to_rgb
from kivy.clock import Clock
from kivy.graphics import Mesh, InstructionGroup, Color
from kivy.logger import Logger
from kivy.properties import (NumericProperty, BoundedNumericProperty,
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, get_hex_from_color
def _update_hex(self, dt):
    try:
        if len(self._upd_hex_list) != 9:
            return
        self._updating_clr = False
        self.hex_color = self._upd_hex_list
    finally:
        self._updating_clr = False
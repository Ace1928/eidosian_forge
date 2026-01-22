from math import cos, sin, pi, sqrt, atan
from colorsys import rgb_to_hsv, hsv_to_rgb
from kivy.clock import Clock
from kivy.graphics import Mesh, InstructionGroup, Color
from kivy.logger import Logger
from kivy.properties import (NumericProperty, BoundedNumericProperty,
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, get_hex_from_color
def _trigger_update_hex(self, text):
    if self._updating_clr:
        return
    self._updating_clr = True
    self._upd_hex_list = text
    ev = self._update_hex_ev
    if ev is None:
        ev = self._update_hex_ev = Clock.create_trigger(self._update_hex)
    ev()
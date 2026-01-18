from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
def update_childsize(self, childs=None):
    width, height = self.size
    if childs is None:
        childs = self.children
    for w in childs:
        shw, shh = w.size_hint
        shw_min, shh_min = w.size_hint_min
        shw_max, shh_max = w.size_hint_max
        if shw is not None and shh is not None:
            c_w = shw * width
            c_h = shh * height
            if shw_min is not None and c_w < shw_min:
                c_w = shw_min
            elif shw_max is not None and c_w > shw_max:
                c_w = shw_max
            if shh_min is not None and c_h < shh_min:
                c_h = shh_min
            elif shh_max is not None and c_h > shh_max:
                c_h = shh_max
            w.size = (c_w, c_h)
        elif shw is not None:
            c_w = shw * width
            if shw_min is not None and c_w < shw_min:
                c_w = shw_min
            elif shw_max is not None and c_w > shw_max:
                c_w = shw_max
            w.width = c_w
        elif shh is not None:
            c_h = shh * height
            if shh_min is not None and c_h < shh_min:
                c_h = shh_min
            elif shh_max is not None and c_h > shh_max:
                c_h = shh_max
            w.height = c_h
        for key, value in w.pos_hint.items():
            if key == 'x':
                w.x = value * width
            elif key == 'right':
                w.right = value * width
            elif key == 'y':
                w.y = value * height
            elif key == 'top':
                w.top = value * height
            elif key == 'center_x':
                w.center_x = value * width
            elif key == 'center_y':
                w.center_y = value * height
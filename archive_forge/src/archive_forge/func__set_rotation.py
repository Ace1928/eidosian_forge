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
def _set_rotation(self, x):
    x = int(x % 360)
    if x == self._rotation:
        return
    if x not in (0, 90, 180, 270):
        raise ValueError('can rotate only 0, 90, 180, 270 degrees')
    self._rotation = x
    if not self.initialized:
        return
    self.dispatch('on_pre_resize', *self.size)
    self.dispatch('on_rotate', x)
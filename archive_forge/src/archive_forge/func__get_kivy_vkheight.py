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
def _get_kivy_vkheight(self):
    mode = Config.get('kivy', 'keyboard_mode')
    if mode in ['dock', 'systemanddock'] and self._vkeyboard_cls is not None:
        for w in self.children:
            if isinstance(w, self._vkeyboard_cls):
                vkeyboard_height = w.height * w.scale
                if self.softinput_mode == 'pan':
                    return vkeyboard_height
                elif self.softinput_mode == 'below_target' and w.target.y < vkeyboard_height:
                    return vkeyboard_height - w.target.y
    return 0
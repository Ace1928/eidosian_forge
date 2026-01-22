from kivy.clock import Clock
from kivy.core import core_select_lib
from kivy.event import EventDispatcher
from kivy.logger import Logger
from kivy.compat import PY2
def _set_volume(self, volume):
    self._volume = volume
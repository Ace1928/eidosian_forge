from kivy.utils import platform
from kivy.event import EventDispatcher
from kivy.logger import Logger
from kivy.core import core_select_lib
def _set_index(self, x):
    if x == self._index:
        return
    self._index = x
    self.init_camera()
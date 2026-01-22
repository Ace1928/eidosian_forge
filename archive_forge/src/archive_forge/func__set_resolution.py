from kivy.utils import platform
from kivy.event import EventDispatcher
from kivy.logger import Logger
from kivy.core import core_select_lib
def _set_resolution(self, res):
    self._resolution = res
    self.init_camera()
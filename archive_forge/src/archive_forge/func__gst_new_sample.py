from gi.repository import Gst
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.camera import CameraBase
from kivy.support import install_gobject_iteration
from kivy.logger import Logger
from ctypes import Structure, c_void_p, c_int, string_at
from weakref import ref
import atexit
def _gst_new_sample(self, *largs):
    sample = self._camerasink.emit('pull-sample')
    if sample is None:
        return False
    self._sample = sample
    if self._texturesize is None:
        for pad in self._decodebin.srcpads:
            s = pad.get_current_caps().get_structure(0)
            self._texturesize = (s.get_value('width'), s.get_value('height'))
            Clock.schedule_once(self._update)
            return False
    Clock.schedule_once(self._update)
    return False
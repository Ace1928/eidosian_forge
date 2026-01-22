from kivy.utils import platform
from kivy.event import EventDispatcher
from kivy.logger import Logger
from kivy.core import core_select_lib
def _copy_to_gpu(self):
    """Copy the buffer into the texture."""
    if self._texture is None:
        Logger.debug('Camera: copy_to_gpu() failed, _texture is None !')
        return
    self._texture.blit_buffer(self._buffer, colorfmt=self._format)
    self._buffer = None
    self.dispatch('on_texture')
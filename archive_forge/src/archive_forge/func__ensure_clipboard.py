from kivy import Logger
from kivy.core import core_select_lib
from kivy.utils import platform
from kivy.setupconfig import USE_SDL2
def _ensure_clipboard(self):
    """ Ensure that the clipboard has been properly initialized.
        """
    if hasattr(self, '_clip_mime_type'):
        return
    if platform == 'win':
        self._clip_mime_type = 'text/plain;charset=utf-8'
        self._encoding = 'utf-16-le'
    elif platform == 'linux':
        self._clip_mime_type = 'text/plain;charset=utf-8'
        self._encoding = 'utf-8'
    else:
        self._clip_mime_type = 'text/plain'
        self._encoding = 'utf-8'
from kivy import Logger
from kivy.core import core_select_lib
from kivy.utils import platform
from kivy.setupconfig import USE_SDL2
def _paste(self):
    self._ensure_clipboard()
    _clip_types = Clipboard.get_types()
    mime_type = self._clip_mime_type
    if mime_type not in _clip_types:
        mime_type = 'text/plain'
    data = self.get(mime_type)
    if data is not None:
        if isinstance(data, bytes):
            data = data.decode(self._encoding, 'ignore')
        data = data.replace(u'\x00', u'')
        return data
    return u''
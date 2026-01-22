from kivy.utils import platform
from kivy.support import install_gobject_iteration
from kivy.core.clipboard import ClipboardBase
import gi
from gi.repository import Gtk, Gdk
class ClipboardGtk3(ClipboardBase):
    _is_init = False

    def init(self):
        if self._is_init:
            return
        install_gobject_iteration()
        self._is_init = True

    def get(self, mimetype='text/plain;charset=utf-8'):
        self.init()
        if mimetype == 'text/plain;charset=utf-8':
            contents = clipboard.wait_for_text()
            if contents:
                return contents
        return ''

    def put(self, data, mimetype='text/plain;charset=utf-8'):
        self.init()
        if mimetype == 'text/plain;charset=utf-8':
            text = data.decode(self._encoding)
            clipboard.set_text(text, -1)
            clipboard.store()

    def get_types(self):
        self.init()
        return ['text/plain;charset=utf-8']
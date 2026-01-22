from kivy.utils import platform
from kivy.core.clipboard import ClipboardBase
class ClipboardDbusKlipper(ClipboardBase):
    _is_init = False

    def init(self):
        if ClipboardDbusKlipper._is_init:
            return
        self.iface = dbus.Interface(proxy, 'org.kde.klipper.klipper')
        ClipboardDbusKlipper._is_init = True

    def get(self, mimetype='text/plain'):
        self.init()
        return str(self.iface.getClipboardContents())

    def put(self, data, mimetype='text/plain'):
        self.init()
        self.iface.setClipboardContents(data.replace('\x00', ''))

    def get_types(self):
        self.init()
        return [u'text/plain']
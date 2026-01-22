import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
@pyglet.window.win32.Win32EventHandler(0)
def _event_wt_packetext(self, msg, wParam, lParam):
    packet = wintab.PACKETEXT()
    if lib.WTPacket(lParam, wParam, ctypes.byref(packet)) == 0:
        return
    if packet.pkBase.nContext == self._context:
        if packet.pkExpKeys.nControl < self.express_key_ct:
            current_state = self.express_keys[packet.pkExpKeys.nControl][packet.pkExpKeys.nLocation]
            new_state = bool(packet.pkExpKeys.nState)
            if current_state != new_state:
                event_type = 'on_express_key_press' if new_state else 'on_express_key_release'
                self.express_keys[packet.pkExpKeys.nControl][packet.pkExpKeys.nLocation] = new_state
                self.dispatch_event(event_type, packet.pkExpKeys.nControl, packet.pkExpKeys.nLocation)
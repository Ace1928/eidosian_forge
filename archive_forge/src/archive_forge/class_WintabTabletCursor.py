import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
class WintabTabletCursor:

    def __init__(self, device, index):
        self.device = device
        self._cursor = wintab.WTI_CURSORS + index
        self.name = wtinfo_string(self._cursor, wintab.CSR_NAME).strip()
        self.active = wtinfo_bool(self._cursor, wintab.CSR_ACTIVE)
        pktdata = wtinfo_wtpkt(self._cursor, wintab.CSR_PKTDATA)
        self.bogus = not (pktdata & wintab.PK_X and pktdata & wintab.PK_Y)
        if self.bogus:
            return
        self.id = wtinfo_dword(self._cursor, wintab.CSR_TYPE) << 32 | wtinfo_dword(self._cursor, wintab.CSR_PHYSID)

    def __repr__(self):
        return 'WintabCursor(%r)' % self.name
import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class CursorNotifyMask:
    DisplayCursor = 1 << 0
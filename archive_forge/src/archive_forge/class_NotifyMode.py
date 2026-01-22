import xcffib
import struct
import io
from . import xfixes
from . import xproto
class NotifyMode:
    Normal = 0
    Grab = 1
    Ungrab = 2
    WhileGrabbed = 3
    PassiveGrab = 4
    PassiveUngrab = 5
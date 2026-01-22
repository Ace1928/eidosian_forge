import xcffib
import struct
import io
from . import xfixes
from . import xproto
class BarrierFlags:
    PointerReleased = 1 << 0
    DeviceIsGrabbed = 1 << 1
import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class ClientDisconnectFlags:
    Default = 0
    Terminate = 1 << 0
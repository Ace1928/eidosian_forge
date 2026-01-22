import xcffib
import struct
import io
from . import xproto
class ClientIdMask:
    ClientXID = 1 << 0
    LocalClientPID = 1 << 1
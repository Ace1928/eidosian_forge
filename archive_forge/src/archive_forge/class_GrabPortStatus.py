import xcffib
import struct
import io
from . import xproto
from . import shm
class GrabPortStatus:
    Success = 0
    BadExtension = 1
    AlreadyGrabbed = 2
    InvalidTime = 3
    BadReply = 4
    BadAlloc = 5
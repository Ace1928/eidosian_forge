import xcffib
import struct
import io
class GrabStatus:
    Success = 0
    AlreadyGrabbed = 1
    InvalidTime = 2
    NotViewable = 3
    Frozen = 4
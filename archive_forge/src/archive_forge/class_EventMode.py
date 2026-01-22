import xcffib
import struct
import io
from . import xfixes
from . import xproto
class EventMode:
    AsyncDevice = 0
    SyncDevice = 1
    ReplayDevice = 2
    AsyncPairedDevice = 3
    AsyncPair = 4
    SyncPair = 5
    AcceptTouch = 6
    RejectTouch = 7
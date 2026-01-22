import xcffib
import struct
import io
from . import xfixes
from . import xproto
class HierarchyChangeType:
    AddMaster = 1
    RemoveMaster = 2
    AttachSlave = 3
    DetachSlave = 4
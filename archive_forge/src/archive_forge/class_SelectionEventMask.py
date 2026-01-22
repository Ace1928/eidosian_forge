import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class SelectionEventMask:
    SetSelectionOwner = 1 << 0
    SelectionWindowDestroy = 1 << 1
    SelectionClientClose = 1 << 2
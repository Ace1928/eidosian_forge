import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class SelectionEvent:
    SetSelectionOwner = 0
    SelectionWindowDestroy = 1
    SelectionClientClose = 2
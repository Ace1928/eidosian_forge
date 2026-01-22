import xcffib
import struct
import io
from . import xproto
class SwapAction:
    Undefined = 0
    Background = 1
    Untouched = 2
    Copied = 3
import xcffib
import struct
import io
from . import xproto
from . import render
class Transform:
    Unit = 1 << 0
    ScaleUp = 1 << 1
    ScaleDown = 1 << 2
    Projective = 1 << 3
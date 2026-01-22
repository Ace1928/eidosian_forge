import xcffib
import struct
import io
from . import xproto
from . import render
class SetConfig:
    Success = 0
    InvalidConfigTime = 1
    InvalidTime = 2
    Failed = 3
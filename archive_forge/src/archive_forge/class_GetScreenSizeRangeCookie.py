import xcffib
import struct
import io
from . import xproto
from . import render
class GetScreenSizeRangeCookie(xcffib.Cookie):
    reply_type = GetScreenSizeRangeReply
import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
class FetchRegionCookie(xcffib.Cookie):
    reply_type = FetchRegionReply
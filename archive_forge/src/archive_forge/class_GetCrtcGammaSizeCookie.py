import xcffib
import struct
import io
from . import xproto
from . import render
class GetCrtcGammaSizeCookie(xcffib.Cookie):
    reply_type = GetCrtcGammaSizeReply
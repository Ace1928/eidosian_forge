import xcffib
import struct
import io
from . import xproto
from . import render
class GetOutputInfoCookie(xcffib.Cookie):
    reply_type = GetOutputInfoReply
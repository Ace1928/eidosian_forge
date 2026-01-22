import xcffib
import struct
import io
from . import xproto
from . import render
class GetScreenInfoCookie(xcffib.Cookie):
    reply_type = GetScreenInfoReply
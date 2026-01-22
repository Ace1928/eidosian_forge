import xcffib
import struct
import io
from . import xproto
from . import render
class GetOutputPropertyCookie(xcffib.Cookie):
    reply_type = GetOutputPropertyReply
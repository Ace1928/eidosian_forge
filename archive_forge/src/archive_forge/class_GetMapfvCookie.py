import xcffib
import struct
import io
from . import xproto
class GetMapfvCookie(xcffib.Cookie):
    reply_type = GetMapfvReply
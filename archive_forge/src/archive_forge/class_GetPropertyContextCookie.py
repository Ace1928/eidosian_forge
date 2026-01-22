import xcffib
import struct
import io
from . import xproto
class GetPropertyContextCookie(xcffib.Cookie):
    reply_type = GetPropertyContextReply
import xcffib
import struct
import io
from . import xproto
class GetMinmaxCookie(xcffib.Cookie):
    reply_type = GetMinmaxReply
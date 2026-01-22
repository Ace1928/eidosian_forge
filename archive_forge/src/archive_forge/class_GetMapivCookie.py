import xcffib
import struct
import io
from . import xproto
class GetMapivCookie(xcffib.Cookie):
    reply_type = GetMapivReply
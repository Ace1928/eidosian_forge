import xcffib
import struct
import io
from . import xproto
class GetBuffersWithFormatCookie(xcffib.Cookie):
    reply_type = GetBuffersWithFormatReply
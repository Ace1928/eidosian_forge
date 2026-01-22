import xcffib
import struct
import io
from . import xproto
class GetCompressedTexImageARBCookie(xcffib.Cookie):
    reply_type = GetCompressedTexImageARBReply
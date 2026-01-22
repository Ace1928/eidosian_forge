import xcffib
import struct
import io
from . import xproto
class GetVersionCookie(xcffib.Cookie):
    reply_type = GetVersionReply
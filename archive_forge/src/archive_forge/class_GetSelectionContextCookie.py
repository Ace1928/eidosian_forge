import xcffib
import struct
import io
from . import xproto
class GetSelectionContextCookie(xcffib.Cookie):
    reply_type = GetSelectionContextReply
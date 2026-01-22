import xcffib
import struct
import io
from . import xproto
class CompareCursorCookie(xcffib.Cookie):
    reply_type = CompareCursorReply
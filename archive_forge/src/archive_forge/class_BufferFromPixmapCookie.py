import xcffib
import struct
import io
from . import xproto
class BufferFromPixmapCookie(xcffib.Cookie):
    reply_type = BufferFromPixmapReply
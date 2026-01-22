import xcffib
import struct
import io
from . import xproto
class MakeContextCurrentCookie(xcffib.Cookie):
    reply_type = MakeContextCurrentReply
import xcffib
import struct
import io
from . import xproto
class MakeCurrentCookie(xcffib.Cookie):
    reply_type = MakeCurrentReply
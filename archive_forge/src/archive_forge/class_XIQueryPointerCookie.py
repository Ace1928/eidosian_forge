import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIQueryPointerCookie(xcffib.Cookie):
    reply_type = XIQueryPointerReply
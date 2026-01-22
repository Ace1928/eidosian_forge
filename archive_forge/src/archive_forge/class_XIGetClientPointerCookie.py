import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIGetClientPointerCookie(xcffib.Cookie):
    reply_type = XIGetClientPointerReply
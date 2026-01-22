import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIListPropertiesCookie(xcffib.Cookie):
    reply_type = XIListPropertiesReply
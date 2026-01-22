import xcffib
import struct
import io
from . import xfixes
from . import xproto
class QueryDeviceStateCookie(xcffib.Cookie):
    reply_type = QueryDeviceStateReply
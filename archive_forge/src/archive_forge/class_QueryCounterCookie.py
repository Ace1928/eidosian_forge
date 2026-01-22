import xcffib
import struct
import io
from . import xproto
class QueryCounterCookie(xcffib.Cookie):
    reply_type = QueryCounterReply
import xcffib
import struct
import io
from . import xproto
class QueryExtentsCookie(xcffib.Cookie):
    reply_type = QueryExtentsReply
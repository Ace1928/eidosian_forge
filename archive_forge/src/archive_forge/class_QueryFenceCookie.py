import xcffib
import struct
import io
from . import xproto
class QueryFenceCookie(xcffib.Cookie):
    reply_type = QueryFenceReply
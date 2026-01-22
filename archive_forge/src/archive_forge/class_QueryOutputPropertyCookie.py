import xcffib
import struct
import io
from . import xproto
from . import render
class QueryOutputPropertyCookie(xcffib.Cookie):
    reply_type = QueryOutputPropertyReply
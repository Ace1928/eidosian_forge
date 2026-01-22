import xcffib
import struct
import io
from . import xproto
from . import render
class QueryProviderPropertyCookie(xcffib.Cookie):
    reply_type = QueryProviderPropertyReply
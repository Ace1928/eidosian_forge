import xcffib
import struct
import io
from . import xproto
from . import shm
class QueryImageAttributesCookie(xcffib.Cookie):
    reply_type = QueryImageAttributesReply
import xcffib
import struct
import io
from . import xproto
class CapableCookie(xcffib.Cookie):
    reply_type = CapableReply
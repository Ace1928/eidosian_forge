import xcffib
import struct
import io
from . import xproto
class GetMSCCookie(xcffib.Cookie):
    reply_type = GetMSCReply
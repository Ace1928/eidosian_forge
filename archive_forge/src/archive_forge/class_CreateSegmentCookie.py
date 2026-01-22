import xcffib
import struct
import io
from . import xproto
class CreateSegmentCookie(xcffib.Cookie):
    reply_type = CreateSegmentReply
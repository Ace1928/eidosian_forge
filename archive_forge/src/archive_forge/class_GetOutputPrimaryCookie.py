import xcffib
import struct
import io
from . import xproto
from . import render
class GetOutputPrimaryCookie(xcffib.Cookie):
    reply_type = GetOutputPrimaryReply
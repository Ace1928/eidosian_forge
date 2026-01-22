import xcffib
import struct
import io
from . import xproto
from . import render
class SetPanningCookie(xcffib.Cookie):
    reply_type = SetPanningReply
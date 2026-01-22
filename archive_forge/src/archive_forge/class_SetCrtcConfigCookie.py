import xcffib
import struct
import io
from . import xproto
from . import render
class SetCrtcConfigCookie(xcffib.Cookie):
    reply_type = SetCrtcConfigReply
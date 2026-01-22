import xcffib
import struct
import io
from . import xproto
from . import render
class GetCrtcGammaCookie(xcffib.Cookie):
    reply_type = GetCrtcGammaReply
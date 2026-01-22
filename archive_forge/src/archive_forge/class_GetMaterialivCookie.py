import xcffib
import struct
import io
from . import xproto
class GetMaterialivCookie(xcffib.Cookie):
    reply_type = GetMaterialivReply
import xcffib
import struct
import io
from . import xproto
class IsTextureCookie(xcffib.Cookie):
    reply_type = IsTextureReply
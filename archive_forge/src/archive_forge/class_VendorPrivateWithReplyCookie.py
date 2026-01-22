import xcffib
import struct
import io
from . import xproto
class VendorPrivateWithReplyCookie(xcffib.Cookie):
    reply_type = VendorPrivateWithReplyReply
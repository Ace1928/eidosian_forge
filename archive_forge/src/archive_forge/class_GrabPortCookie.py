import xcffib
import struct
import io
from . import xproto
from . import shm
class GrabPortCookie(xcffib.Cookie):
    reply_type = GrabPortReply
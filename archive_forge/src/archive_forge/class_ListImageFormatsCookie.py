import xcffib
import struct
import io
from . import xproto
from . import shm
class ListImageFormatsCookie(xcffib.Cookie):
    reply_type = ListImageFormatsReply
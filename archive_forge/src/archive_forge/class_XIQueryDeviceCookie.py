import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIQueryDeviceCookie(xcffib.Cookie):
    reply_type = XIQueryDeviceReply
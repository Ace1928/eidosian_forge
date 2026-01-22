import ctypes
from pyglet.libs.win32 import com
def DIDFT_GETINSTANCE(n):
    return n >> 8 & 65535
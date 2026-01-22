import socket
import os
import sys
class in_addr(ctypes.Structure):
    _fields_ = [('S_addr', ctypes.c_ubyte * 4)]
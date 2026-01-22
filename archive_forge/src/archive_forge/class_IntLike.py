from ctypes import *
import unittest
import struct
class IntLike:

    def __int__(self):
        return 2
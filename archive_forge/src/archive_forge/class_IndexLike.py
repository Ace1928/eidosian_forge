from ctypes import *
import unittest
import struct
class IndexLike:

    def __index__(self):
        return 2
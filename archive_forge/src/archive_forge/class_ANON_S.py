import unittest
import test.support
from ctypes import *
class ANON_S(Structure):
    _fields_ = [('a', c_int)]
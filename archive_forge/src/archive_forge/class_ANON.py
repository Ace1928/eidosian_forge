import unittest
import test.support
from ctypes import *
class ANON(Union):
    _fields_ = [('a', c_int), ('b', c_int)]
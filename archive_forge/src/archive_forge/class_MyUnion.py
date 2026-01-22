from ctypes import *
from ctypes.test import need_symbol
import unittest
import sys
class MyUnion(Union):
    _fields_ = [('a', c_int)]
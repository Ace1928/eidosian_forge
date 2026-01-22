import unittest
from test.support import bigmemtest, _2G
import sys
from ctypes import *
from ctypes.test import need_symbol
class EmptyArray(Array):
    _type_ = c_int
    _length_ = 0
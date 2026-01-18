import unittest
from ctypes import *
from ctypes.test import need_symbol
def test_param_1(self):
    BUF = c_char * 4
    buf = BUF()
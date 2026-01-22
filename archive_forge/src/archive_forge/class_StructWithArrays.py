import unittest
from ctypes import *
import re, sys
class StructWithArrays(Structure):
    _fields_ = [('x', c_long * 3 * 2), ('y', Point * 4)]
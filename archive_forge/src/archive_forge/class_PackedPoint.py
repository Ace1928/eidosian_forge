import unittest
from ctypes import *
import re, sys
class PackedPoint(Structure):
    _pack_ = 2
    _fields_ = [('x', c_long), ('y', c_long)]
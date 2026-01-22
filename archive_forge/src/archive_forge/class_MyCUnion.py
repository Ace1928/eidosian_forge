import unittest
from ctypes import *
class MyCUnion(Union):
    _fields_ = (('field', c_int),)
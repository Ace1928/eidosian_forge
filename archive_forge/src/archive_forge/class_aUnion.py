import unittest
from ctypes import *
import re, sys
class aUnion(Union):
    _fields_ = [('a', c_int)]
import unittest
from ctypes import *
from sys import getrefcount as grc
def assertSame(self, a, b):
    self.assertEqual(id(a), id(b))
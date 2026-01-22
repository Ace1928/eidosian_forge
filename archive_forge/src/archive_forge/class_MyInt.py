import unittest
from ctypes import *
class MyInt(c_int):

    def __eq__(self, other):
        if type(other) != MyInt:
            return NotImplementedError
        return self.value == other.value
import unittest
from ctypes import *
import _ctypes_test
class BadSequence(tuple):

    def __getitem__(self, key):
        if key == 0:
            return 'my_strchr'
        if key == 1:
            return CDLL(_ctypes_test.__file__)
        raise IndexError
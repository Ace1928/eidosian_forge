import sys
import ctypes
from ctypes import *
import unittest
def _addr_at(self, key):
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) != self.nd:
        raise ValueError('wrong number of indexes')
    for i in range(self.nd):
        if not 0 <= key[i] < self.shape[i]:
            raise IndexError(f'index {i} out of range')
    return self.data + sum((i * s for i, s in zip(key, self.strides)))
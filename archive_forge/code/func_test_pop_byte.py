import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def test_pop_byte(self):
    l = List(self, 4, 0)
    values = [b'aaaa', b'bbbb', b'cccc', b'dddd', b'eeee', b'ffff', b'gggg', b'hhhhh']
    for i in values:
        l.append(i)
    self.assertEqual(len(l), 8)
    received = l.pop()
    self.assertEqual(b'hhhh', received)
    self.assertEqual(len(l), 7)
    received = [j for j in l]
    self.assertEqual(received, values[:-1])
    received = l.pop(0)
    self.assertEqual(b'aaaa', received)
    self.assertEqual(len(l), 6)
    received = l.pop(2)
    self.assertEqual(b'dddd', received)
    self.assertEqual(len(l), 5)
    expected = [b'bbbb', b'cccc', b'eeee', b'ffff', b'gggg']
    received = [j for j in l]
    self.assertEqual(received, expected)
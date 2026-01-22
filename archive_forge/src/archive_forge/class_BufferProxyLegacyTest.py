import re
import weakref
import gc
import ctypes
import unittest
import pygame
from pygame.bufferproxy import BufferProxy
class BufferProxyLegacyTest(unittest.TestCase):
    content = b'\x01\x00\x00\x02' * 12
    buffer = ctypes.create_string_buffer(content)
    data = (ctypes.addressof(buffer), True)

    def test_length(self):
        bf = BufferProxy({'shape': (3, 4), 'typestr': '|u4', 'data': self.data, 'strides': (12, 4)})
        self.assertEqual(bf.length, len(self.content))
        bf = BufferProxy({'shape': (3, 3), 'typestr': '|u4', 'data': self.data, 'strides': (12, 4)})
        self.assertEqual(bf.length, 3 * 3 * 4)

    def test_raw(self):
        bf = BufferProxy({'shape': (len(self.content),), 'typestr': '|u1', 'data': self.data})
        self.assertEqual(bf.raw, self.content)
        bf = BufferProxy({'shape': (3, 4), 'typestr': '|u4', 'data': self.data, 'strides': (4, 12)})
        self.assertEqual(bf.raw, self.content)
        bf = BufferProxy({'shape': (3, 4), 'typestr': '|u1', 'data': self.data, 'strides': (16, 4)})
        self.assertRaises(ValueError, getattr, bf, 'raw')

    def test_write(self):
        from ctypes import c_byte, sizeof, addressof, string_at, memset
        nullbyte = b'\x00'
        Buf = c_byte * 10
        data_buf = Buf(*range(1, 3 * sizeof(Buf) + 1, 3))
        data = string_at(data_buf, sizeof(data_buf))
        buf = Buf()
        bp = BufferProxy({'typestr': '|u1', 'shape': (sizeof(buf),), 'data': (addressof(buf), False)})
        try:
            self.assertEqual(bp.raw, nullbyte * sizeof(Buf))
            bp.write(data)
            self.assertEqual(bp.raw, data)
            memset(buf, 0, sizeof(buf))
            bp.write(data[:3], 2)
            raw = bp.raw
            self.assertEqual(raw[:2], nullbyte * 2)
            self.assertEqual(raw[2:5], data[:3])
            self.assertEqual(raw[5:], nullbyte * (sizeof(Buf) - 5))
            bp.write(data[:3], bp.length - 3)
            raw = bp.raw
            self.assertEqual(raw[-3:], data[:3])
            self.assertRaises(IndexError, bp.write, data, 1)
            self.assertRaises(IndexError, bp.write, data[:5], -1)
            self.assertRaises(IndexError, bp.write, data[:5], bp.length)
            self.assertRaises(TypeError, bp.write, 12)
            bp = BufferProxy({'typestr': '|u1', 'shape': (sizeof(buf),), 'data': (addressof(buf), True)})
            self.assertRaises(pygame.BufferError, bp.write, b'123')
        finally:
            bp = None
            gc.collect()
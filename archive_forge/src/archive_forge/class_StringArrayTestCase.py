import unittest
from ctypes import *
from ctypes.test import need_symbol
class StringArrayTestCase(unittest.TestCase):

    def test(self):
        BUF = c_char * 4
        buf = BUF(b'a', b'b', b'c')
        self.assertEqual(buf.value, b'abc')
        self.assertEqual(buf.raw, b'abc\x00')
        buf.value = b'ABCD'
        self.assertEqual(buf.value, b'ABCD')
        self.assertEqual(buf.raw, b'ABCD')
        buf.value = b'x'
        self.assertEqual(buf.value, b'x')
        self.assertEqual(buf.raw, b'x\x00CD')
        buf[1] = b'Z'
        self.assertEqual(buf.value, b'xZCD')
        self.assertEqual(buf.raw, b'xZCD')
        self.assertRaises(ValueError, setattr, buf, 'value', b'aaaaaaaa')
        self.assertRaises(TypeError, setattr, buf, 'value', 42)

    def test_c_buffer_value(self):
        buf = c_buffer(32)
        buf.value = b'Hello, World'
        self.assertEqual(buf.value, b'Hello, World')
        self.assertRaises(TypeError, setattr, buf, 'value', memoryview(b'Hello, World'))
        self.assertRaises(TypeError, setattr, buf, 'value', memoryview(b'abc'))
        self.assertRaises(ValueError, setattr, buf, 'raw', memoryview(b'x' * 100))

    def test_c_buffer_raw(self):
        buf = c_buffer(32)
        buf.raw = memoryview(b'Hello, World')
        self.assertEqual(buf.value, b'Hello, World')
        self.assertRaises(TypeError, setattr, buf, 'value', memoryview(b'abc'))
        self.assertRaises(ValueError, setattr, buf, 'raw', memoryview(b'x' * 100))

    def test_param_1(self):
        BUF = c_char * 4
        buf = BUF()

    def test_param_2(self):
        BUF = c_char * 4
        buf = BUF()

    def test_del_segfault(self):
        BUF = c_char * 4
        buf = BUF()
        with self.assertRaises(AttributeError):
            del buf.raw
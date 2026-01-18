import platform
from platform import architecture as _architecture
import struct
import sys
import unittest
from ctypes.test import need_symbol
from ctypes import (CDLL, Array, Structure, Union, POINTER, sizeof, byref, alignment,
from ctypes.util import find_library
from struct import calcsize
import _ctypes_test
from collections import namedtuple
from test import support
@unittest.skipIf(True, 'Test disabled for now - see bpo-16575/bpo-16576')
def test_union_by_value(self):

    class Nested1(Structure):
        _fields_ = [('an_int', c_int), ('another_int', c_int)]

    class Test4(Union):
        _fields_ = [('a_long', c_long), ('a_struct', Nested1)]

    class Nested2(Structure):
        _fields_ = [('an_int', c_int), ('a_union', Test4)]

    class Test5(Structure):
        _fields_ = [('an_int', c_int), ('nested', Nested2), ('another_int', c_int)]
    test4 = Test4()
    dll = CDLL(_ctypes_test.__file__)
    with self.assertRaises(TypeError) as ctx:
        func = dll._testfunc_union_by_value1
        func.restype = c_long
        func.argtypes = (Test4,)
        result = func(test4)
    self.assertEqual(ctx.exception.args[0], 'item 1 in _argtypes_ passes a union by value, which is unsupported.')
    test5 = Test5()
    with self.assertRaises(TypeError) as ctx:
        func = dll._testfunc_union_by_value2
        func.restype = c_long
        func.argtypes = (Test5,)
        result = func(test5)
    self.assertEqual(ctx.exception.args[0], 'item 1 in _argtypes_ passes a union by value, which is unsupported.')
    test4.a_long = 12345
    func = dll._testfunc_union_by_reference1
    func.restype = c_long
    func.argtypes = (POINTER(Test4),)
    result = func(byref(test4))
    self.assertEqual(result, 12345)
    self.assertEqual(test4.a_long, 0)
    self.assertEqual(test4.a_struct.an_int, 0)
    self.assertEqual(test4.a_struct.another_int, 0)
    test4.a_struct.an_int = 305397760
    test4.a_struct.another_int = 22136
    func = dll._testfunc_union_by_reference2
    func.restype = c_long
    func.argtypes = (POINTER(Test4),)
    result = func(byref(test4))
    self.assertEqual(result, 305419896)
    self.assertEqual(test4.a_long, 0)
    self.assertEqual(test4.a_struct.an_int, 0)
    self.assertEqual(test4.a_struct.another_int, 0)
    test5.an_int = 301989888
    test5.nested.an_int = 3429888
    test5.another_int = 120
    func = dll._testfunc_union_by_reference3
    func.restype = c_long
    func.argtypes = (POINTER(Test5),)
    result = func(byref(test5))
    self.assertEqual(result, 305419896)
    self.assertEqual(test5.an_int, 0)
    self.assertEqual(test5.nested.an_int, 0)
    self.assertEqual(test5.another_int, 0)
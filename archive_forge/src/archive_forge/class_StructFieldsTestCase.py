import unittest
from ctypes import *
class StructFieldsTestCase(unittest.TestCase):

    def test_1_A(self):

        class X(Structure):
            pass
        self.assertEqual(sizeof(X), 0)
        X._fields_ = []
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_1_B(self):

        class X(Structure):
            _fields_ = []
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_2(self):

        class X(Structure):
            pass
        X()
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_3(self):

        class X(Structure):
            pass

        class Y(Structure):
            _fields_ = [('x', X)]
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_4(self):

        class X(Structure):
            pass

        class Y(X):
            pass
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])
        Y._fields_ = []
        self.assertRaises(AttributeError, setattr, X, '_fields_', [])

    def test_5(self):

        class X(Structure):
            _fields_ = (('char', c_char * 5),)
        x = X(b'#' * 5)
        x.char = b'a\x00b\x00'
        self.assertEqual(bytes(x), b'a\x00###')

    def test_6(self):

        class X(Structure):
            _fields_ = [('x', c_int)]
        CField = type(X.x)
        self.assertRaises(TypeError, CField)

    def test_gh99275(self):

        class BrokenStructure(Structure):

            def __init_subclass__(cls, **kwargs):
                cls._fields_ = []
        with self.assertRaisesRegex(TypeError, 'ctypes state is not initialized'):

            class Subclass(BrokenStructure):
                ...

    def test___set__(self):

        class MyCStruct(Structure):
            _fields_ = (('field', c_int),)
        self.assertRaises(TypeError, MyCStruct.field.__set__, 'wrong type self', 42)

        class MyCUnion(Union):
            _fields_ = (('field', c_int),)
        self.assertRaises(TypeError, MyCUnion.field.__set__, 'wrong type self', 42)

    def test___get__(self):

        class MyCStruct(Structure):
            _fields_ = (('field', c_int),)
        self.assertRaises(TypeError, MyCStruct.field.__get__, 'wrong type self', 42)

        class MyCUnion(Union):
            _fields_ = (('field', c_int),)
        self.assertRaises(TypeError, MyCUnion.field.__get__, 'wrong type self', 42)
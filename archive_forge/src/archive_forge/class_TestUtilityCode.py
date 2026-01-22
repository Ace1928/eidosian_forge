import unittest
from Cython.Compiler import Code, UtilityCode
class TestUtilityCode(unittest.TestCase):

    def test_equality(self):
        c1 = Code.UtilityCode.load('NumpyImportUFunc', 'NumpyImportArray.c')
        c2 = Code.UtilityCode.load('NumpyImportArray', 'NumpyImportArray.c')
        c3 = Code.UtilityCode.load('pyunicode_strlen', 'StringTools.c')
        c4 = Code.UtilityCode.load('pyunicode_from_unicode', 'StringTools.c')
        c5 = Code.UtilityCode.load('IncludeStringH', 'StringTools.c')
        c6 = Code.UtilityCode.load('IncludeCppStringH', 'StringTools.c')
        codes = [c1, c2, c3, c4, c5, c6]
        for m in range(len(codes)):
            for n in range(len(codes)):
                if n == m:
                    self.assertEqual(codes[m], codes[n])
                else:
                    self.assertNotEqual(codes[m], codes[n])
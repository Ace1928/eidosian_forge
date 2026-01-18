import sys
import unittest
from Cython.Utils import (
def test_normalise_float_repr(self):
    examples = [('.0', '.0'), ('.000000', '.0'), ('.1', '.1'), ('1.', '1.'), ('1.0', '1.'), ('1.000000000000000000000', '1.'), ('00000000000000000000001.000000000000000000000', '1.'), ('12345.0025', '12345.0025'), ('1E5', '100000.'), ('.1E-5', '.000001'), ('1.1E-5', '.000011'), ('12.3E-5', '.000123'), ('.1E10', '1000000000.'), ('1.1E10', '11000000000.'), ('123.4E10', '1234000000000.'), ('123.456E0', '123.456'), ('123.456E-1', '12.3456'), ('123.456E-2', '1.23456'), ('123.456E1', '1234.56'), ('123.456E2', '12345.6'), ('2.1E80', '210000000000000000000000000000000000000000000000000000000000000000000000000000000.')]
    for float_str, norm_str in examples:
        self.assertEqual(float(float_str), float(norm_str))
        result = normalise_float_repr(float_str)
        self.assertEqual(float(float_str), float(result))
        self.assertEqual(result, norm_str, 'normalise_float_repr(%r) == %r != %r  (%.330f)' % (float_str, result, norm_str, float(float_str)))
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, DataShapeSyntaxError
def test_primitive_measure_repr(self):
    self.assertEqual(repr(datashape.int8), 'ctype("int8")')
    self.assertEqual(repr(datashape.int16), 'ctype("int16")')
    self.assertEqual(repr(datashape.int32), 'ctype("int32")')
    self.assertEqual(repr(datashape.int64), 'ctype("int64")')
    self.assertEqual(repr(datashape.uint8), 'ctype("uint8")')
    self.assertEqual(repr(datashape.uint16), 'ctype("uint16")')
    self.assertEqual(repr(datashape.uint32), 'ctype("uint32")')
    self.assertEqual(repr(datashape.uint64), 'ctype("uint64")')
    self.assertEqual(repr(datashape.float32), 'ctype("float32")')
    self.assertEqual(repr(datashape.float64), 'ctype("float64")')
    self.assertEqual(repr(datashape.string), 'ctype("string")')
    self.assertEqual(repr(datashape.String(3)), 'ctype("string[3]")')
    self.assertEqual(repr(datashape.String('A')), 'ctype("string[\'A\']")')
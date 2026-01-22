from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
class TestDataShapeParseTuple(unittest.TestCase):

    def setUp(self):
        self.sym = datashape.TypeSymbolTable()

    def test_tuple(self):
        self.assertEqual(parse('(float32)', self.sym), ct.DataShape(ct.Tuple([ct.DataShape(ct.float32)])))
        self.assertEqual(parse('(int16, int32)', self.sym), ct.DataShape(ct.Tuple([ct.DataShape(ct.int16), ct.DataShape(ct.int32)])))
        self.assertEqual(parse('(float32,)', self.sym), ct.DataShape(ct.Tuple([ct.DataShape(ct.float32)])))
        self.assertEqual(parse('(int16, int32,)', self.sym), ct.DataShape(ct.Tuple([ct.DataShape(ct.int16), ct.DataShape(ct.int32)])))
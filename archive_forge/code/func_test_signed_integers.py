from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_signed_integers(self):
    self.assertEqual(parse('int8', self.sym), ct.DataShape(ct.int8))
    self.assertEqual(parse('int16', self.sym), ct.DataShape(ct.int16))
    self.assertEqual(parse('int32', self.sym), ct.DataShape(ct.int32))
    self.assertEqual(parse('int64', self.sym), ct.DataShape(ct.int64))
    self.assertEqual(parse('int', self.sym), ct.DataShape(ct.int_))
    self.assertEqual(parse('int', self.sym), parse('int32', self.sym))
    self.assertEqual(parse('intptr', self.sym), ct.DataShape(ct.intptr))
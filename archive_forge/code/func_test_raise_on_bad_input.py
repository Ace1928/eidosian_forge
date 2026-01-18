from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_raise_on_bad_input(self):
    self.assertRaises(TypeError, dshape, None)
    self.assertRaises(TypeError, dshape, lambda x: x + 1)
    self.assertRaises(datashape.DataShapeSyntaxError, dshape, '1 *')
    self.assertRaises(datashape.DataShapeSyntaxError, dshape, '1,')
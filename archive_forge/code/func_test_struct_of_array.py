from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_struct_of_array(self):
    self.assertEqual(str(dshape('5 * int32')), '5 * int32')
    self.assertEqual(str(dshape('{field: 5 * int32}')), '{field: 5 * int32}')
    self.assertEqual(str(dshape('{field: M * int32}')), '{field: M * int32}')
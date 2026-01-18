from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_python_containers(self):
    var = datashape.Var()
    int32 = datashape.int32
    self.assertEqual(dshape('3 * int32'), dshape((3, int32)))
    self.assertEqual(dshape('3 * int32'), dshape([3, int32]))
    self.assertEqual(dshape('var * 3 * int32'), dshape((var, 3, int32)))
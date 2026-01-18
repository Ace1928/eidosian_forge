from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_ragged_array(self):
    self.assertTrue(isinstance(dshape('3 * var * int32')[1], datashape.Var))
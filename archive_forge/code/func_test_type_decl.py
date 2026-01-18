from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
@pytest.mark.xfail(reason='type decl has been removed in the new parser')
def test_type_decl(self):
    self.assertRaises(error.DataShapeTypeError, dshape, 'type X T = 3, T')
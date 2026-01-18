import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, DataShapeSyntaxError
def test_structure_repr(self):
    self.assertEqual(repr(dshape('{x:int32, y:int64}')), 'dshape("{x: int32, y: int64}")')
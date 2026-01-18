import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_setslice():
    vec = ri.IntSexpVector([1, 2, 3])
    vec[0:2] = ri.IntSexpVector([11, 12])
    assert len(vec) == 3
    assert vec[0] == 11
    assert vec[1] == 12
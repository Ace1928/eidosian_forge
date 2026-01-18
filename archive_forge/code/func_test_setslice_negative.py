import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_setslice_negative():
    vec = ri.IntSexpVector([1, 2, 3])
    vec[-2:-1] = ri.IntSexpVector([33])
    assert len(vec) == 3
    assert vec[1] == 33
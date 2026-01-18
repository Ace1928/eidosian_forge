from graphql import Undefined
from ..scalars import Boolean, Float, Int, String
def test_serializes_output_int():
    assert Int.serialize(1) == 1
    assert Int.serialize(0) == 0
    assert Int.serialize(-1) == -1
    assert Int.serialize(0.1) == 0
    assert Int.serialize(1.1) == 1
    assert Int.serialize(-1.1) == -1
    assert Int.serialize(100000.0) == 100000
    assert Int.serialize(9876504321) is Undefined
    assert Int.serialize(-9876504321) is Undefined
    assert Int.serialize(1e+100) is Undefined
    assert Int.serialize(-1e+100) is Undefined
    assert Int.serialize('-1.1') == -1
    assert Int.serialize('one') is Undefined
    assert Int.serialize(False) == 0
    assert Int.serialize(True) == 1
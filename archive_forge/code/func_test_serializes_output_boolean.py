from graphql import Undefined
from ..scalars import Boolean, Float, Int, String
def test_serializes_output_boolean():
    assert Boolean.serialize('string') is True
    assert Boolean.serialize('') is False
    assert Boolean.serialize(1) is True
    assert Boolean.serialize(0) is False
    assert Boolean.serialize(True) is True
    assert Boolean.serialize(False) is False
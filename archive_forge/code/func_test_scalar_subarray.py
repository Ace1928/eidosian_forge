from datashader import datashape
import pytest
def test_scalar_subarray():
    assert datashape.int32.subarray(0) == datashape.int32
    with pytest.raises(IndexError):
        datashape.int32.subarray(1)
    assert datashape.string.subarray(0) == datashape.string
    with pytest.raises(IndexError):
        datashape.string.subarray(1)
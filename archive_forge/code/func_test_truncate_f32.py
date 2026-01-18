import pytest
from numpy.core._simd import targets
def test_truncate_f32(self):
    if not npyv.simd_f32:
        pytest.skip("F32 isn't support by the SIMD extension")
    f32 = npyv.setall_f32(0.1)[0]
    assert f32 != 0.1
    assert round(f32, 1) == 0.1
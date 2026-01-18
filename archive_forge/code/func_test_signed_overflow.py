import pytest
from numpy.core._simd import targets
@pytest.mark.parametrize('sfx', signed_sfx)
def test_signed_overflow(self, sfx):
    nlanes = getattr(npyv, 'nlanes_' + sfx)
    maxs_72 = (1 << 71) - 1
    lane = getattr(npyv, 'setall_' + sfx)(maxs_72)[0]
    assert lane == -1
    lanes = getattr(npyv, 'load_' + sfx)([maxs_72] * nlanes)
    assert lanes == [-1] * nlanes
    mins_72 = -1 << 71
    lane = getattr(npyv, 'setall_' + sfx)(mins_72)[0]
    assert lane == 0
    lanes = getattr(npyv, 'load_' + sfx)([mins_72] * nlanes)
    assert lanes == [0] * nlanes
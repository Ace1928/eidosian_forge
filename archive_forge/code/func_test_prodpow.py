from .._util import prodpow
from ..util.testing import requires
def test_prodpow():
    result = prodpow([11, 13], [[0, 1], [1, 2]])
    assert result[0] == 13
    assert result[1] == 11 * 13 * 13
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.period import (
import pandas._testing as tm
class TestExtractOrdinals:

    def test_extract_ordinals_raises(self):
        arr = np.arange(5)
        freq = to_offset('D')
        with pytest.raises(TypeError, match='values must be object-dtype'):
            extract_ordinals(arr, freq)

    def test_extract_ordinals_2d(self):
        freq = to_offset('D')
        arr = np.empty(10, dtype=object)
        arr[:] = iNaT
        res = extract_ordinals(arr, freq)
        res2 = extract_ordinals(arr.reshape(5, 2), freq)
        tm.assert_numpy_array_equal(res, res2.reshape(-1))
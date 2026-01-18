import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('allow_fill', [True, False])
def test_take_out_of_bounds_raises(self, data, allow_fill):
    arr = data[:3]
    with pytest.raises(IndexError, match='out of bounds|out-of-bounds'):
        arr.take(np.asarray([0, 3]), allow_fill=allow_fill)
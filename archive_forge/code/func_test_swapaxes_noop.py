import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_swapaxes_noop(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    msg = "'DataFrame.swapaxes' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        tm.assert_frame_equal(df, df.swapaxes(0, 0))
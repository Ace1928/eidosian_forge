import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_swapaxes_invalid_axis(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    msg = "'DataFrame.swapaxes' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            df.swapaxes(2, 5)
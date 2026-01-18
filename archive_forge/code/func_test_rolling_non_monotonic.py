from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('method,expected', [('var', [float('nan'), 43.0, float('nan'), 136.333333, 43.5, 94.966667, 182.0, 318.0]), ('mean', [float('nan'), 7.5, float('nan'), 21.5, 6.0, 9.166667, 13.0, 17.5]), ('sum', [float('nan'), 30.0, float('nan'), 86.0, 30.0, 55.0, 91.0, 140.0]), ('skew', [float('nan'), 0.709296, float('nan'), 0.407073, 0.984656, 0.919184, 0.874674, 0.842418]), ('kurt', [float('nan'), -0.5916711736073559, float('nan'), -1.0028993131317954, -0.06103844629409494, -0.254143227116194, -0.37362637362637585, -0.45439658241367054])])
def test_rolling_non_monotonic(method, expected):
    """
    Make sure the (rare) branch of non-monotonic indices is covered by a test.

    output from 1.1.3 is assumed to be the expected output. Output of sum/mean has
    manually been verified.

    GH 36933.
    """
    use_expanding = [True, False, True, False, True, True, True, True]
    df = DataFrame({'values': np.arange(len(use_expanding)) ** 2})

    class CustomIndexer(BaseIndexer):

        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = i + 1
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return (start, end)
    indexer = CustomIndexer(window_size=4, use_expanding=use_expanding)
    result = getattr(df.rolling(indexer), method)()
    expected = DataFrame({'values': expected})
    tm.assert_frame_equal(result, expected)
from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,expected', [({'a': [1, 2, 'a'], 'b': [4, 5, 6]}, DataFrame({'b': [1.0, 2.0, 3.0]}, columns=Index(['b'], dtype=object))), ({'a': [1, 2, 'a']}, DataFrame(index=range(3), columns=[]))])
def test_rank_mixed_axis_zero(self, data, expected):
    df = DataFrame(data, columns=Index(list(data.keys()), dtype=object))
    with pytest.raises(TypeError, match="'<' not supported between instances of"):
        df.rank()
    result = df.rank(numeric_only=True)
    tm.assert_frame_equal(result, expected)
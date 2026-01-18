import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('id_vars, value_vars, col_level, expected', [(['A'], ['B'], 0, DataFrame({'A': {0: 1.067683, 1: -1.321405, 2: -0.807333}, 'CAP': {0: 'B', 1: 'B', 2: 'B'}, 'value': {0: -1.110463, 1: 0.368915, 2: 0.08298}})), (['a'], ['b'], 1, DataFrame({'a': {0: 1.067683, 1: -1.321405, 2: -0.807333}, 'low': {0: 'b', 1: 'b', 2: 'b'}, 'value': {0: -1.110463, 1: 0.368915, 2: 0.08298}}))])
def test_single_vars_work_with_multiindex(self, id_vars, value_vars, col_level, expected, df1):
    result = df1.melt(id_vars, value_vars, col_level=col_level)
    tm.assert_frame_equal(result, expected)
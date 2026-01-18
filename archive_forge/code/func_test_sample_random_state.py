import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('func_str,arg', [('np.array', [2, 3, 1, 0]), ('np.random.MT19937', 3), ('np.random.PCG64', 11)])
def test_sample_random_state(self, func_str, arg, frame_or_series):
    obj = DataFrame({'col1': range(10, 20), 'col2': range(20, 30)})
    obj = tm.get_obj(obj, frame_or_series)
    result = obj.sample(n=3, random_state=eval(func_str)(arg))
    expected = obj.sample(n=3, random_state=com.random_state(eval(func_str)(arg)))
    tm.assert_equal(result, expected)
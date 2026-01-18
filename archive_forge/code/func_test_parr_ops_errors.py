import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('ng', ['str', 1.5])
@pytest.mark.parametrize('func', [lambda obj, ng: obj + ng, lambda obj, ng: ng + obj, lambda obj, ng: obj - ng, lambda obj, ng: ng - obj, lambda obj, ng: np.add(obj, ng), lambda obj, ng: np.add(ng, obj), lambda obj, ng: np.subtract(obj, ng), lambda obj, ng: np.subtract(ng, obj)])
def test_parr_ops_errors(self, ng, func, box_with_array):
    idx = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq='M', name='idx')
    obj = tm.box_expected(idx, box_with_array)
    msg = '|'.join(['unsupported operand type\\(s\\)', 'can only concatenate', 'must be str', 'object to str implicitly'])
    with pytest.raises(TypeError, match=msg):
        func(obj, ng)
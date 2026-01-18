import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('other', [Categorical(['b', 'a']), Categorical(['b', 'a'], categories=['b', 'a'], ordered=True), Categorical(['b', 'a'], categories=['a', 'b', 'c'], ordered=True)])
def test_setitem_same_ordered_raises(self, other):
    target = Categorical(['a', 'b'], categories=['a', 'b'], ordered=True)
    mask = np.array([True, False])
    msg = 'Cannot set a Categorical with another, without identical categories'
    with pytest.raises(TypeError, match=msg):
        target[mask] = other[mask]
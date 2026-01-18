import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_other_type_raises(self):
    msg = 'dtype bool cannot be converted to timedelta64\\[ns\\]'
    with pytest.raises(TypeError, match=msg):
        TimedeltaArray._from_sequence(np.array([1, 2, 3], dtype='bool'))
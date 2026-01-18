import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_upsampling_without_replacement(self, frame_or_series):
    obj = DataFrame({'A': list('abc')})
    obj = tm.get_obj(obj, frame_or_series)
    msg = 'Replace has to be set to `True` when upsampling the population `frac` > 1.'
    with pytest.raises(ValueError, match=msg):
        obj.sample(frac=2, replace=False)
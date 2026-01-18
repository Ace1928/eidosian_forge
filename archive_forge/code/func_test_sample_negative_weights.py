import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_negative_weights(self, obj):
    bad_weights = [-0.1] * 10
    msg = 'weight vector many not include negative values'
    with pytest.raises(ValueError, match=msg):
        obj.sample(n=3, weights=bad_weights)
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_missing_weights(self, obj):
    nan_weights = [np.nan] * 10
    with pytest.raises(ValueError, match='Invalid weights: weights sum to zero'):
        obj.sample(n=3, weights=nan_weights)
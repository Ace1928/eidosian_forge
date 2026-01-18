import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_transpose_non_default_axes(index_or_series_obj):
    msg = "the 'axes' parameter is not supported"
    obj = index_or_series_obj
    with pytest.raises(ValueError, match=msg):
        obj.transpose(1)
    with pytest.raises(ValueError, match=msg):
        obj.transpose(axes=1)
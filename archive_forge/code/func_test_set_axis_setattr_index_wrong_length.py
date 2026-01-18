import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_axis_setattr_index_wrong_length(self, obj):
    msg = f'Length mismatch: Expected axis has {len(obj)} elements, new values have {len(obj) - 1} elements'
    with pytest.raises(ValueError, match=msg):
        obj.index = np.arange(len(obj) - 1)
    if obj.ndim == 2:
        with pytest.raises(ValueError, match='Length mismatch'):
            obj.columns = obj.columns[::2]
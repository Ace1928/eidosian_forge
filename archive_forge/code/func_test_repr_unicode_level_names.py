from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_unicode_level_names(self, frame_or_series):
    index = MultiIndex.from_tuples([(0, 0), (1, 1)], names=['Δ', 'i1'])
    obj = DataFrame(np.random.randn(2, 4), index=index)
    obj = tm.get_obj(obj, frame_or_series)
    repr(obj)
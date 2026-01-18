from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('op_wrapper', [lambda x: x, lambda x: [x], lambda x: {'A': x}, lambda x: {'A': [x]}])
def test_transform_reducer_raises(all_reductions, frame_or_series, op_wrapper):
    op = op_wrapper(all_reductions)
    obj = DataFrame({'A': [1, 2, 3]})
    obj = tm.get_obj(obj, frame_or_series)
    msg = 'Function did not transform'
    with pytest.raises(ValueError, match=msg):
        obj.transform(op)
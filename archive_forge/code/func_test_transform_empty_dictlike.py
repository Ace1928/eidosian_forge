import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('ops', [{}, {'A': []}, {'A': [], 'B': 'cumsum'}, {'A': 'cumsum', 'B': []}, {'A': [], 'B': ['cumsum']}, {'A': ['cumsum'], 'B': []}])
def test_transform_empty_dictlike(float_frame, ops, frame_or_series):
    obj = unpack_obj(float_frame, frame_or_series, 0)
    with pytest.raises(ValueError, match='No transform functions were provided'):
        obj.transform(ops)
from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_validate_ragged_array_fastpath():
    start_indices = np.array([0, 2, 5, 6, 6, 11], dtype='uint16')
    flat_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='float32')
    valid_dict = dict(start_indices=start_indices, flat_array=flat_array)
    RaggedArray(valid_dict)
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, start_indices=25))
    ve.match('start_indices property of a RaggedArray')
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, start_indices=start_indices.astype('float32')))
    ve.match('start_indices property of a RaggedArray')
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, start_indices=np.array([start_indices])))
    ve.match('start_indices property of a RaggedArray')
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, flat_array='foo'))
    ve.match('flat_array property of a RaggedArray')
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, flat_array=np.array([flat_array])))
    ve.match('flat_array property of a RaggedArray')
    bad_start_indices = start_indices.copy()
    bad_start_indices[-1] = 99
    with pytest.raises(ValueError) as ve:
        RaggedArray(dict(valid_dict, start_indices=bad_start_indices))
    ve.match('start_indices must be less than')
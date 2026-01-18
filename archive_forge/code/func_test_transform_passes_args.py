import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('use_apply', [True, False])
def test_transform_passes_args(use_apply, frame_or_series):
    expected_args = [1, 2]
    expected_kwargs = {'c': 3}

    def f(x, a, b, c):
        if use_apply == isinstance(x, frame_or_series):
            raise ValueError
        assert [a, b] == expected_args
        assert c == expected_kwargs['c']
        return x
    frame_or_series([1]).transform(f, 0, *expected_args, **expected_kwargs)
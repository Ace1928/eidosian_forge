from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_nuisance_depr_passes_through_warnings():

    def expected_warning(x):
        warnings.warn('Hello, World!')
        return x.sum()
    df = DataFrame({'a': [1, 2, 3]})
    with tm.assert_produces_warning(UserWarning, match='Hello, World!'):
        df.agg([expected_warning])
from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
@pytest.mark.slow
def test_pivot_number_of_levels_larger_than_int32(self, monkeypatch):

    class MockUnstacker(reshape_lib._Unstacker):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            raise Exception("Don't compute final result.")
    with monkeypatch.context() as m:
        m.setattr(reshape_lib, '_Unstacker', MockUnstacker)
        df = DataFrame({'ind1': np.arange(2 ** 16), 'ind2': np.arange(2 ** 16), 'count': 0})
        msg = 'The following operation may generate'
        with tm.assert_produces_warning(PerformanceWarning, match=msg):
            with pytest.raises(Exception, match="Don't compute final result."):
                df.pivot_table(index='ind1', columns='ind2', values='count', aggfunc='count')
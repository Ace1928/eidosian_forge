from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_option', [True, 'bad', 1])
@pytest.mark.parametrize('ties_method', ['average', 'min', 'max', 'first', 'dense'])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('pct', [True, False])
@pytest.mark.parametrize('vals', [['bar', 'bar', 'foo', 'bar', 'baz'], ['bar', np.nan, 'foo', np.nan, 'baz'], [1, np.nan, 2, np.nan, 3]])
def test_rank_naoption_raises(ties_method, ascending, na_option, pct, vals):
    df = DataFrame({'key': ['foo'] * 5, 'val': vals})
    msg = "na_option must be one of 'keep', 'top', or 'bottom'"
    with pytest.raises(ValueError, match=msg):
        df.groupby('key').rank(method=ties_method, ascending=ascending, na_option=na_option, pct=pct)
import numpy as np
import pytest
from pandas.compat import IS64
import pandas as pd
import pandas._testing as tm
@pytest.mark.skipif(not IS64, reason='GH 36579: fail on 32-bit system')
@pytest.mark.parametrize('pandasmethname, kwargs', [('var', {'ddof': 0}), ('var', {'ddof': 1}), ('std', {'ddof': 0}), ('std', {'ddof': 1}), ('kurtosis', {}), ('skew', {}), ('sem', {})])
def test_stat_method(pandasmethname, kwargs):
    s = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, np.nan, np.nan], dtype='Float64')
    pandasmeth = getattr(s, pandasmethname)
    result = pandasmeth(**kwargs)
    s2 = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype='float64')
    pandasmeth = getattr(s2, pandasmethname)
    expected = pandasmeth(**kwargs)
    assert expected == result
import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('method', [operator.methodcaller('total_seconds')])
def test_timedelta_methods(method):
    s = pd.Series(pd.timedelta_range('2000', periods=4))
    s.attrs = {'a': 1}
    result = method(s.dt)
    assert result.attrs == {'a': 1}
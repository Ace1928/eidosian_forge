import numpy as np
import pytest
import pandas as pd
@pytest.mark.parametrize('dropna', [True, False])
def test_reductions_return_types(dropna, data, all_numeric_reductions):
    op = all_numeric_reductions
    s = pd.Series(data)
    if dropna:
        s = s.dropna()
    if op in ('sum', 'prod'):
        assert isinstance(getattr(s, op)(), np.int_)
    elif op == 'count':
        assert isinstance(getattr(s, op)(), np.integer)
    elif op in ('min', 'max'):
        assert isinstance(getattr(s, op)(), np.bool_)
    else:
        assert isinstance(getattr(s, op)(), np.float64)
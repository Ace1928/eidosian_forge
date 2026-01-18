import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_numeric_like_ops(self):
    df = DataFrame({'value': np.random.default_rng(2).integers(0, 10000, 100)})
    labels = [f'{i} - {i + 499}' for i in range(0, 10000, 500)]
    cat_labels = Categorical(labels, labels)
    df = df.sort_values(by=['value'], ascending=True)
    df['value_group'] = pd.cut(df.value, range(0, 10500, 500), right=False, labels=cat_labels)
    for op, str_rep in [('__add__', '\\+'), ('__sub__', '-'), ('__mul__', '\\*'), ('__truediv__', '/')]:
        msg = f'Series cannot perform the operation {str_rep}|unsupported operand'
        with pytest.raises(TypeError, match=msg):
            getattr(df, op)(df)
    s = df['value_group']
    for op in ['kurt', 'skew', 'var', 'std', 'mean', 'sum', 'median']:
        msg = f"does not support reduction '{op}'"
        with pytest.raises(TypeError, match=msg):
            getattr(s, op)(numeric_only=False)
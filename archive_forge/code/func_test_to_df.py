import warnings
import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_to_df():
    import pandas
    import modin.pandas as pd
    from modin.tests.pandas.utils import df_equals
    modin_df = pd.DataFrame(np.array([1, 2, 3]))
    pandas_df = pandas.DataFrame(numpy.array([1, 2, 3]))
    df_equals(pandas_df, modin_df)
    modin_df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))
    pandas_df = pandas.DataFrame(numpy.array([[1, 2, 3], [4, 5, 6]]))
    df_equals(pandas_df, modin_df)
    for kw in [{}, {'dtype': str}]:
        modin_df, pandas_df = [lib[0].DataFrame(lib[1].array([[1, 2, 3], [4, 5, 6]]), columns=['col 0', 'col 1', 'col 2'], index=pd.Index([4, 6]), **kw) for lib in ((pd, np), (pandas, numpy))]
        df_equals(pandas_df, modin_df)
    df_equals(pandas_df, modin_df)
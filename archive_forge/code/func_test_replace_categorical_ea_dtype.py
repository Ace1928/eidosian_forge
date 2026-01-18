import pytest
import pandas as pd
from pandas import Categorical
import pandas._testing as tm
def test_replace_categorical_ea_dtype():
    cat = Categorical(pd.array(['a', 'b'], dtype='string'))
    msg = 'The behavior of Series\\.replace \\(and DataFrame.replace\\) with CategoricalDtype'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = pd.Series(cat).replace(['a', 'b'], ['c', pd.NA])._values
    expected = Categorical(pd.array(['c', pd.NA], dtype='string'))
    tm.assert_categorical_equal(result, expected)
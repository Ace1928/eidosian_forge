import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_pandas_Categorical_accessors():
    if not have_pandas_categorical:
        return
    c = pandas_Categorical_from_codes([1, 1, 0, -1], ['a', 'b'])
    assert np.all(pandas_Categorical_categories(c) == ['a', 'b'])
    assert np.all(pandas_Categorical_codes(c) == [1, 1, 0, -1])
    if have_pandas_categorical_dtype:
        s = pandas.Series(c)
        assert np.all(pandas_Categorical_categories(s) == ['a', 'b'])
        assert np.all(pandas_Categorical_codes(s) == [1, 1, 0, -1])
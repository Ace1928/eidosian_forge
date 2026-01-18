import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_safe_is_pandas_categorical():
    assert not safe_is_pandas_categorical(np.arange(10))
    if have_pandas_categorical:
        c_obj = pandas.Categorical(['a', 'b'])
        assert safe_is_pandas_categorical(c_obj)
    if have_pandas_categorical_dtype:
        s_obj = pandas.Series(['a', 'b'], dtype='category')
        assert safe_is_pandas_categorical(s_obj)
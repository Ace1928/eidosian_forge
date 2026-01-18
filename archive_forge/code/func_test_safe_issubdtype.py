import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_safe_issubdtype():
    assert safe_issubdtype(int, np.integer)
    assert safe_issubdtype(np.dtype(float), np.floating)
    assert not safe_issubdtype(int, np.floating)
    assert not safe_issubdtype(np.dtype(float), np.integer)
    if have_pandas_categorical_dtype:
        bad_dtype = pandas.Series(['a', 'b'], dtype='category')
        assert not safe_issubdtype(bad_dtype, np.integer)
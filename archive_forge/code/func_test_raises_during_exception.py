import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_raises_during_exception():
    msg = "Did not see expected warning of class 'UserWarning'"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(UserWarning):
            raise ValueError
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(UserWarning):
            warnings.warn('FutureWarning', FutureWarning)
            raise IndexError
    msg = 'Caused unexpected warning'
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(None):
            warnings.warn('FutureWarning', FutureWarning)
            raise SystemError
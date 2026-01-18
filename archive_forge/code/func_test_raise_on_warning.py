import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_raise_on_warning(self, false_or_none):
    msg = 'Caused unexpected warning\\(s\\)'
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(false_or_none):
            f()
import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_raise_on_extra_warning(pair_different_warnings):
    expected_category, extra_category = pair_different_warnings
    match = 'Caused unexpected warning\\(s\\)'
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn('Expected warning', expected_category)
            warnings.warn('Unexpected warning NOT OK', extra_category)
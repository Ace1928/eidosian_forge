import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_same_category_different_messages_first_match():
    category = UserWarning
    with tm.assert_produces_warning(category, match='^Match this'):
        warnings.warn('Match this', category)
        warnings.warn('Do not match that', category)
        warnings.warn('Do not match that either', category)
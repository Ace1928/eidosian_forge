from copy import (
import pytest
from pandas import MultiIndex
import pandas._testing as tm
def test_shallow_copy(idx):
    i_copy = idx._view()
    assert_multiindex_copied(i_copy, idx)
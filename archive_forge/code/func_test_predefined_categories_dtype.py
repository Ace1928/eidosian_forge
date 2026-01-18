import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_predefined_categories_dtype():
    """Check that the categories_ dtype is `object` for string categories

    Regression test for gh-25171.
    """
    categories = [['as', 'mmas', 'eas', 'ras', 'acs'], ['1', '2']]
    enc = OneHotEncoder(categories=categories)
    enc.fit([['as', '1']])
    assert len(categories) == len(enc.categories_)
    for n, cat in enumerate(enc.categories_):
        assert cat.dtype == object
        assert_array_equal(categories[n], cat)
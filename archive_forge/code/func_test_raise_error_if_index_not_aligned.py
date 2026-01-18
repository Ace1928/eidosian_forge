import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_raise_error_if_index_not_aligned():
    """Check column transformer raises error if indices are not aligned.

    Non-regression test for gh-26210.
    """
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame([[1.0, 2.2], [3.0, 1.0]], columns=['a', 'b'], index=[8, 3])
    reset_index_transformer = FunctionTransformer(lambda x: x.reset_index(drop=True), feature_names_out='one-to-one')
    ct = ColumnTransformer([('num1', 'passthrough', ['a']), ('num2', reset_index_transformer, ['b'])])
    ct.set_output(transform='pandas')
    msg = "Concatenating DataFrames from the transformer's output lead to an inconsistent number of samples. The output may have Pandas Indexes that do not match."
    with pytest.raises(ValueError, match=msg):
        ct.fit_transform(X)
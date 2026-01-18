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
def test_remainder_set_output():
    """Check that the output is set for the remainder.

    Non-regression test for #26306.
    """
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'a': [True, False, True], 'b': [1, 2, 3]})
    ct = make_column_transformer((VarianceThreshold(), make_column_selector(dtype_include=bool)), remainder=VarianceThreshold(), verbose_feature_names_out=False)
    ct.set_output(transform='pandas')
    out = ct.fit_transform(df)
    pd.testing.assert_frame_equal(out, df)
    ct.set_output(transform='default')
    out = ct.fit_transform(df)
    assert isinstance(out, np.ndarray)
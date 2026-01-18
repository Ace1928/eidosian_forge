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
@pytest.mark.parametrize('transformers, remainder, expected_names', [([('bycol1', TransWithNames(), ['d', 'c']), ('bycol2', 'passthrough', ['a'])], 'passthrough', ['d', 'c', 'a', 'b']), ([('bycol1', TransWithNames(['a']), ['d', 'c']), ('bycol2', 'passthrough', ['d'])], 'drop', ['a', 'd']), ([('bycol1', TransWithNames(), ['b']), ('bycol2', 'drop', ['d'])], 'passthrough', ['b', 'a', 'c']), ([('bycol1', TransWithNames(['pca1', 'pca2']), ['a', 'b', 'd'])], 'passthrough', ['pca1', 'pca2', 'c']), ([('bycol1', TransWithNames(['a', 'c']), ['d']), ('bycol2', 'passthrough', ['d'])], 'drop', ['a', 'c', 'd']), ([('bycol1', TransWithNames([f'pca{i}' for i in range(2)]), ['b']), ('bycol2', TransWithNames([f'kpca{i}' for i in range(2)]), ['b'])], 'passthrough', ['pca0', 'pca1', 'kpca0', 'kpca1', 'a', 'c', 'd']), ([('bycol1', 'drop', ['d'])], 'drop', []), ([('bycol1', TransWithNames(), slice(1, 2)), ('bycol2', 'drop', ['d'])], 'passthrough', ['b', 'a', 'c']), ([('bycol1', TransWithNames(), ['b']), ('bycol2', 'drop', slice(3, 4))], 'passthrough', ['b', 'a', 'c']), ([('bycol1', TransWithNames(), ['d', 'c']), ('bycol2', 'passthrough', slice(0, 2))], 'drop', ['d', 'c', 'a', 'b']), ([('bycol1', TransWithNames(), slice('a', 'b')), ('bycol2', 'drop', ['d'])], 'passthrough', ['a', 'b', 'c']), ([('bycol1', TransWithNames(), ['b']), ('bycol2', 'drop', slice('c', 'd'))], 'passthrough', ['b', 'a']), ([('bycol1', TransWithNames(), ['d', 'c']), ('bycol2', 'passthrough', slice('a', 'b'))], 'drop', ['d', 'c', 'a', 'b']), ([('bycol1', TransWithNames(), ['d', 'c']), ('bycol2', 'passthrough', slice('b', 'b'))], 'drop', ['d', 'c', 'b'])])
def test_verbose_feature_names_out_false(transformers, remainder, expected_names):
    """Check feature_names_out for verbose_feature_names_out=False"""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd'])
    ct = ColumnTransformer(transformers, remainder=remainder, verbose_feature_names_out=False)
    ct.fit(df)
    names = ct.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected_names)
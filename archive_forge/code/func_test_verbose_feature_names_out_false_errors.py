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
@pytest.mark.parametrize('transformers, remainder, colliding_columns', [([('bycol1', TransWithNames(), ['b']), ('bycol2', 'passthrough', ['b'])], 'drop', "['b']"), ([('bycol1', TransWithNames(['c', 'd']), ['c']), ('bycol2', 'passthrough', ['c'])], 'drop', "['c']"), ([('bycol1', TransWithNames(['a']), ['b']), ('bycol2', 'passthrough', ['b'])], 'passthrough', "['a']"), ([('bycol1', TransWithNames(['a']), ['b']), ('bycol2', 'drop', ['b'])], 'passthrough', "['a']"), ([('bycol1', TransWithNames(['c', 'b']), ['b']), ('bycol2', 'passthrough', ['c', 'b'])], 'drop', "['b', 'c']"), ([('bycol1', TransWithNames(['a']), ['b']), ('bycol2', 'passthrough', ['a']), ('bycol3', TransWithNames(['a']), ['b'])], 'passthrough', "['a']"), ([('bycol1', TransWithNames(['a', 'b']), ['b']), ('bycol2', 'passthrough', ['a']), ('bycol3', TransWithNames(['b']), ['c'])], 'passthrough', "['a', 'b']"), ([('bycol1', TransWithNames([f'pca{i}' for i in range(6)]), ['b']), ('bycol2', TransWithNames([f'pca{i}' for i in range(6)]), ['b'])], 'passthrough', "['pca0', 'pca1', 'pca2', 'pca3', 'pca4', ...]"), ([('bycol1', TransWithNames(['a', 'b']), slice(1, 2)), ('bycol2', 'passthrough', ['a']), ('bycol3', TransWithNames(['b']), ['c'])], 'passthrough', "['a', 'b']"), ([('bycol1', TransWithNames(['a', 'b']), ['b']), ('bycol2', 'passthrough', slice(0, 1)), ('bycol3', TransWithNames(['b']), ['c'])], 'passthrough', "['a', 'b']"), ([('bycol1', TransWithNames(['a', 'b']), slice('b', 'c')), ('bycol2', 'passthrough', ['a']), ('bycol3', TransWithNames(['b']), ['c'])], 'passthrough', "['a', 'b']"), ([('bycol1', TransWithNames(['a', 'b']), ['b']), ('bycol2', 'passthrough', slice('a', 'a')), ('bycol3', TransWithNames(['b']), ['c'])], 'passthrough', "['a', 'b']")])
def test_verbose_feature_names_out_false_errors(transformers, remainder, colliding_columns):
    """Check feature_names_out for verbose_feature_names_out=False"""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd'])
    ct = ColumnTransformer(transformers, remainder=remainder, verbose_feature_names_out=False)
    ct.fit(df)
    msg = re.escape(f'Output feature names: {colliding_columns} are not unique. Please set verbose_feature_names_out=True to add prefixes to feature names')
    with pytest.raises(ValueError, match=msg):
        ct.get_feature_names_out()
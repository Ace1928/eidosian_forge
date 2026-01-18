import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('func', [pairwise_distances, pairwise_kernels])
def test_pairwise_precomputed(func):
    with pytest.raises(ValueError, match='.* shape .*'):
        func(np.zeros((5, 3)), metric='precomputed')
    with pytest.raises(ValueError, match='.* shape .*'):
        func(np.zeros((5, 3)), np.zeros((4, 4)), metric='precomputed')
    with pytest.raises(ValueError, match='.* shape .*'):
        func(np.zeros((5, 3)), np.zeros((4, 3)), metric='precomputed')
    S = np.zeros((5, 5))
    S2 = func(S, metric='precomputed')
    assert S is S2
    S = np.zeros((5, 3))
    S2 = func(S, np.zeros((3, 3)), metric='precomputed')
    assert S is S2
    S = func(np.array([[1]], dtype='int'), metric='precomputed')
    assert 'f' == S.dtype.kind
    S = func([[1.0]], metric='precomputed')
    assert isinstance(S, np.ndarray)
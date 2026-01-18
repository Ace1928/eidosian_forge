from itertools import product
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import config_context, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import (
def test_type_of_target_pandas_sparse():
    pd = pytest.importorskip('pandas')
    y = pd.arrays.SparseArray([1, np.nan, np.nan, 1, np.nan])
    msg = "y cannot be class 'SparseSeries' or 'SparseArray'"
    with pytest.raises(ValueError, match=msg):
        type_of_target(y)
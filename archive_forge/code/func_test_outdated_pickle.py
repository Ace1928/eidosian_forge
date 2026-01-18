from functools import partial
from unittest.mock import patch
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets.tests.test_common import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import assert_allclose_dense_sparse
def test_outdated_pickle(fetch_20newsgroups_vectorized_fxt):
    with patch('os.path.exists') as mock_is_exist:
        with patch('joblib.load') as mock_load:
            mock_is_exist.return_value = True
            mock_load.return_value = ('X', 'y')
            err_msg = 'The cached dataset located in'
            with pytest.raises(ValueError, match=err_msg):
                fetch_20newsgroups_vectorized_fxt(as_frame=True)
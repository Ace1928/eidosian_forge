import gzip
import json
import os
import re
from functools import partial
from importlib import resources
from io import BytesIO
from urllib.error import HTTPError
import numpy as np
import pytest
import scipy.sparse
import sklearn
from sklearn import config_context
from sklearn.datasets import fetch_openml as fetch_openml_orig
from sklearn.datasets._openml import (
from sklearn.utils import Bunch, check_pandas_support
from sklearn.utils._testing import (
def test_retry_with_clean_cache_http_error(tmpdir):
    data_id = 61
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))

    @_retry_with_clean_cache(openml_path, cache_directory)
    def _load_data():
        raise HTTPError(url=None, code=412, msg='Simulated mock error', hdrs=None, fp=BytesIO())
    error_msg = 'Simulated mock error'
    with pytest.raises(HTTPError, match=error_msg):
        _load_data()
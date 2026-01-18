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
@pytest.mark.parametrize('gzip_response', [True, False])
def test_open_openml_url_cache(monkeypatch, gzip_response, tmpdir):
    data_id = 61
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    response1 = _open_openml_url(openml_path, cache_directory)
    location = _get_local_path(openml_path, cache_directory)
    assert os.path.isfile(location)
    response2 = _open_openml_url(openml_path, cache_directory)
    assert response1.read() == response2.read()
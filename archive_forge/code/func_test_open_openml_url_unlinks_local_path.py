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
@pytest.mark.parametrize('write_to_disk', [True, False])
def test_open_openml_url_unlinks_local_path(monkeypatch, tmpdir, write_to_disk):
    data_id = 61
    openml_path = sklearn.datasets._openml._DATA_FILE.format(data_id)
    cache_directory = str(tmpdir.mkdir('scikit_learn_data'))
    location = _get_local_path(openml_path, cache_directory)

    def _mock_urlopen(request, *args, **kwargs):
        if write_to_disk:
            with open(location, 'w') as f:
                f.write('')
        raise ValueError('Invalid request')
    monkeypatch.setattr(sklearn.datasets._openml, 'urlopen', _mock_urlopen)
    with pytest.raises(ValueError, match='Invalid request'):
        _open_openml_url(openml_path, cache_directory)
    assert not os.path.exists(location)
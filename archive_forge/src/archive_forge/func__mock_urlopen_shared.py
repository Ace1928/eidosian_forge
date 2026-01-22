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
def _mock_urlopen_shared(url, has_gzip_header, expected_prefix, suffix):
    assert url.startswith(expected_prefix)
    data_file_name = _file_name(url, suffix)
    data_file_path = resources.files(data_module) / data_file_name
    with data_file_path.open('rb') as f:
        if has_gzip_header and gzip_response:
            fp = BytesIO(f.read())
            return _MockHTTPResponse(fp, True)
        else:
            decompressed_f = read_fn(f, 'rb')
            fp = BytesIO(decompressed_f.read())
            return _MockHTTPResponse(fp, False)
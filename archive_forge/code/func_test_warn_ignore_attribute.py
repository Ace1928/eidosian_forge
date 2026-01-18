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
def test_warn_ignore_attribute(monkeypatch, gzip_response):
    data_id = 40966
    expected_row_id_msg = "target_column='{}' has flag is_row_identifier."
    expected_ignore_msg = "target_column='{}' has flag is_ignore."
    _monkey_patch_webbased_functions(monkeypatch, data_id, gzip_response)
    target_col = 'MouseID'
    msg = expected_row_id_msg.format(target_col)
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, target_column=target_col, cache=False, as_frame=False, parser='liac-arff')
    target_col = 'Genotype'
    msg = expected_ignore_msg.format(target_col)
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, target_column=target_col, cache=False, as_frame=False, parser='liac-arff')
    target_col = 'MouseID'
    msg = expected_row_id_msg.format(target_col)
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, target_column=[target_col, 'class'], cache=False, as_frame=False, parser='liac-arff')
    target_col = 'Genotype'
    msg = expected_ignore_msg.format(target_col)
    with pytest.warns(UserWarning, match=msg):
        fetch_openml(data_id=data_id, target_column=[target_col, 'class'], cache=False, as_frame=False, parser='liac-arff')
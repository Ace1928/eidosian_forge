import builtins
import platform
import sys
from contextlib import suppress
from functools import wraps
from os import environ
from unittest import SkipTest
import joblib
import numpy as np
import pytest
from _pytest.doctest import DoctestItem
from threadpoolctl import threadpool_limits
from sklearn import config_context, set_config
from sklearn._min_dependencies import PYTEST_MIN_VERSION
from sklearn.datasets import (
from sklearn.tests import random_seed
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import get_pytest_filterwarning_lines
from sklearn.utils.fixes import (
def _fetch_fixture(f):
    """Fetch dataset (download if missing and requested by environment)."""
    download_if_missing = environ.get('SKLEARN_SKIP_NETWORK_TESTS', '1') == '0'

    @wraps(f)
    def wrapped(*args, **kwargs):
        kwargs['download_if_missing'] = download_if_missing
        try:
            return f(*args, **kwargs)
        except OSError as e:
            if str(e) != 'Data not found and `download_if_missing` is False':
                raise
            pytest.skip('test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0')
    return pytest.fixture(lambda: wrapped)
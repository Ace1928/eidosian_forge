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
class _MockHTTPResponse:

    def __init__(self, data, is_gzip):
        self.data = data
        self.is_gzip = is_gzip

    def read(self, amt=-1):
        return self.data.read(amt)

    def close(self):
        self.data.close()

    def info(self):
        if self.is_gzip:
            return {'Content-Encoding': 'gzip'}
        return {}

    def __iter__(self):
        return iter(self.data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
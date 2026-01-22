import gzip
import os
import shutil
from bz2 import BZ2File
from importlib import resources
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
import sklearn
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, load_svmlight_files
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def _load_svmlight_local_test_file(filename, **kwargs):
    """
    Helper to load resource `filename` with `importlib.resources`
    """
    data_path = _svmlight_local_test_file_path(filename)
    with data_path.open('rb') as f:
        return load_svmlight_file(f, **kwargs)
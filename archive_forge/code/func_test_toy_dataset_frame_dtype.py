import os
import shutil
import tempfile
import warnings
from functools import partial
from importlib import resources
from pathlib import Path
from pickle import dumps, loads
import numpy as np
import pytest
from sklearn.datasets import (
from sklearn.datasets._base import (
from sklearn.datasets.tests.test_common import check_as_frame
from sklearn.preprocessing import scale
from sklearn.utils import Bunch
@pytest.mark.parametrize('loader_func, data_dtype, target_dtype', [(load_breast_cancer, np.float64, int), (load_diabetes, np.float64, np.float64), (load_digits, np.float64, int), (load_iris, np.float64, int), (load_linnerud, np.float64, np.float64), (load_wine, np.float64, int)])
def test_toy_dataset_frame_dtype(loader_func, data_dtype, target_dtype):
    default_result = loader_func()
    check_as_frame(default_result, loader_func, expected_data_dtype=data_dtype, expected_target_dtype=target_dtype)
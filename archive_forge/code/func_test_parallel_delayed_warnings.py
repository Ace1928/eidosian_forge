import time
import joblib
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import config_context, get_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed
def test_parallel_delayed_warnings():
    """Informative warnings should be raised when mixing sklearn and joblib API"""
    warn_msg = '`sklearn.utils.parallel.Parallel` needs to be used in conjunction'
    with pytest.warns(UserWarning, match=warn_msg) as records:
        Parallel()((joblib.delayed(time.sleep)(0) for _ in range(10)))
    assert len(records) == 10
    warn_msg = '`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate'
    with pytest.warns(UserWarning, match=warn_msg) as records:
        joblib.Parallel()((delayed(time.sleep)(0) for _ in range(10)))
    assert len(records) == 10
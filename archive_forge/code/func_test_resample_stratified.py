import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_resample_stratified():
    rng = np.random.RandomState(0)
    n_samples = 100
    p = 0.9
    X = rng.normal(size=(n_samples, 1))
    y = rng.binomial(1, p, size=n_samples)
    _, y_not_stratified = resample(X, y, n_samples=10, random_state=0, stratify=None)
    assert np.all(y_not_stratified == 1)
    _, y_stratified = resample(X, y, n_samples=10, random_state=0, stratify=y)
    assert not np.all(y_stratified == 1)
    assert np.sum(y_stratified) == 9
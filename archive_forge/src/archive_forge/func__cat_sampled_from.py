import gc
import importlib.util
import multiprocessing
import os
import platform
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from platform import system
from typing import (
import numpy as np
import pytest
from scipy import sparse
import xgboost as xgb
from xgboost.core import ArrayLike
from xgboost.sklearn import SklObjective
from xgboost.testing.data import (
from hypothesis import strategies
from hypothesis.extra.numpy import arrays
def _cat_sampled_from() -> strategies.SearchStrategy:

    @strategies.composite
    def _make_cat(draw: Callable) -> Tuple[int, int, int, float]:
        n_samples = draw(strategies.integers(2, 512))
        n_features = draw(strategies.integers(1, 4))
        n_cats = draw(strategies.integers(1, 128))
        sparsity = draw(strategies.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False, allow_subnormal=False))
        return (n_samples, n_features, n_cats, sparsity)

    def _build(args: Tuple[int, int, int, float]) -> TestDataset:
        n_samples = args[0]
        n_features = args[1]
        n_cats = args[2]
        sparsity = args[3]
        return TestDataset(f'{n_samples}x{n_features}-{n_cats}-{sparsity}', lambda: make_categorical(n_samples, n_features, n_cats, False, sparsity), 'reg:squarederror', 'rmse')
    return _make_cat().map(_build)
from __future__ import annotations
import contextlib
from collections.abc import Container, Iterable, Sequence
from functools import wraps
from numbers import Integral
import numpy as np
from tlz import concat
from dask.core import flatten
def argtopk_preprocess(a, idx):
    """Preparatory step for argtopk

    Put data together with its original indices in a tuple.
    """
    return (a, idx)
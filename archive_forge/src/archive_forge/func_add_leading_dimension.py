from __future__ import annotations
import os
from glob import glob
from dask.array.core import Array
from dask.base import tokenize
def add_leading_dimension(x):
    return x[None, ...]
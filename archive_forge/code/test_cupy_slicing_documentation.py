from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.utils import assert_eq
Test that when the indices are a dask array
    they are not accidentally computed
    
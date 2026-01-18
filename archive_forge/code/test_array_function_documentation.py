from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
Test proper dispatch on binary NumPy functions
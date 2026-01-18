from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def test_result_name() -> None:

    class Named:

        def __init__(self, name=None):
            self.name = name
    assert result_name([1, 2]) is None
    assert result_name([Named()]) is None
    assert result_name([Named('foo'), 2]) == 'foo'
    assert result_name([Named('foo'), Named('bar')]) is None
    assert result_name([Named('foo'), Named()]) is None
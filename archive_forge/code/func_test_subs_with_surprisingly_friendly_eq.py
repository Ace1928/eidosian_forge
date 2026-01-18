from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_subs_with_surprisingly_friendly_eq():
    try:
        import pandas as pd
    except ImportError:
        return
    else:
        df = pd.DataFrame()
        assert subs(df, 'x', 1) is df
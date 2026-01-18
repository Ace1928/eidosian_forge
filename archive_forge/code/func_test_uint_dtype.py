import datetime as dt
from unittest import SkipTest, skipIf
import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from numpy import nan
from packaging.version import Version
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import apply_when
from holoviews.streams import Tap
from holoviews.util import render
import logging
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32])
def test_uint_dtype(dtype):
    df = pd.DataFrame(np.arange(2, dtype=dtype), columns=['A'])
    curve = Curve(df)
    img = rasterize(curve, dynamic=False, height=10, width=10)
    assert (np.asarray(img.data['Count']) == np.eye(10)).all()
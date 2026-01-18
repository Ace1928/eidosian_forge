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
def test_rasterize_apply_when_instance_with_line_width(self):
    df = pd.DataFrame(np.random.multivariate_normal((0, 0), [[0.1, 0.1], [0.1, 1.0]], (100,)))
    df.columns = ['a', 'b']
    curve = Curve(df, kdims=['a'], vdims=['b'])
    custom_rasterize = rasterize.instance(line_width=2)
    assert {'line_width': 2} == custom_rasterize._rasterize__instance_kwargs
    output = apply_when(curve, operation=custom_rasterize, predicate=lambda x: len(x) > 10)
    render(output, 'bokeh')
    assert isinstance(output, DynamicMap)
    overlay = output.items()[0][1]
    assert isinstance(overlay, Overlay)
    assert len(overlay) == 2
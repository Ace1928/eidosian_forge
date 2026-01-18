import sys
from unittest import SkipTest
from parameterized import parameterized
import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import pytest
from holoviews import Store, render
from holoviews.element import Image, QuadMesh, Points
from holoviews.core.spaces import DynamicMap
from holoviews.core.overlay import Overlay
from holoviews.element.chart import Scatter
from holoviews.element.comparison import ComparisonTestCase
from hvplot.converter import HoloViewsConverter
from hvplot.tests.util import makeTimeDataFrame
from packaging.version import Version
@parameterized.expand([('rasterize',), ('datashade',)])
def test_operation_resample_when(self, operation):
    df = pd.DataFrame(np.random.multivariate_normal((0, 0), [[0.1, 0.1], [0.1, 1.0]], (5000,))).rename({0: 'x', 1: 'y'}, axis=1)
    dmap = df.hvplot.scatter('x', 'y', resample_when=1000, **{operation: True})
    assert isinstance(dmap, DynamicMap)
    render(dmap)
    overlay = dmap.items()[0][1]
    assert isinstance(overlay, Overlay)
    image = overlay.get(0)
    assert isinstance(image, Image)
    assert len(image.data) > 0
    scatter = overlay.get(1)
    assert isinstance(scatter, Scatter)
    assert len(scatter.data) == 0
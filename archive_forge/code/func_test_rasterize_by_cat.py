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
def test_rasterize_by_cat(self):
    from datashader.reductions import count_cat
    dmap = self.df.hvplot.scatter('x', 'y', by='category', rasterize=True)
    agg = dmap.callback.inputs[0].callback.operation.p.aggregator
    self.assertIsInstance(agg, count_cat)
    self.assertEqual(agg.column, 'category')
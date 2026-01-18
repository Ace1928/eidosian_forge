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
def test_shade_categorical_images_grid(self):
    xs, ys = ([0.25, 0.75], [0.25, 0.75])
    data = NdOverlay({'A': Image((xs, ys, np.array([[1, 0], [0, 0]], dtype='u4')), datatype=['grid'], vdims=Dimension('z Count', nodata=0)), 'B': Image((xs, ys, np.array([[0, 0], [1, 0]], dtype='u4')), datatype=['grid'], vdims=Dimension('z Count', nodata=0)), 'C': Image((xs, ys, np.array([[0, 0], [1, 0]], dtype='u4')), datatype=['grid'], vdims=Dimension('z Count', nodata=0))}, kdims=['z'])
    shaded = shade(data, rescale_discrete_levels=False)
    r = [[228, 120], [66, 120]]
    g = [[26, 109], [150, 109]]
    b = [[28, 95], [129, 95]]
    a = [[40, 0], [255, 0]]
    expected = RGB((xs, ys, r, g, b, a), datatype=['grid'], vdims=RGB.vdims + [Dimension('A', range=(0, 1))])
    self.assertEqual(shaded, expected)
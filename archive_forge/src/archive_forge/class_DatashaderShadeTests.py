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
class DatashaderShadeTests(ComparisonTestCase):

    def test_shade_categorical_images_xarray(self):
        xs, ys = ([0.25, 0.75], [0.25, 0.75])
        data = NdOverlay({'A': Image((xs, ys, np.array([[1, 0], [0, 0]], dtype='u4')), datatype=['xarray'], vdims=Dimension('z Count', nodata=0)), 'B': Image((xs, ys, np.array([[0, 0], [1, 0]], dtype='u4')), datatype=['xarray'], vdims=Dimension('z Count', nodata=0)), 'C': Image((xs, ys, np.array([[0, 0], [1, 0]], dtype='u4')), datatype=['xarray'], vdims=Dimension('z Count', nodata=0))}, kdims=['z'])
        shaded = shade(data, rescale_discrete_levels=False)
        r = [[228, 120], [66, 120]]
        g = [[26, 109], [150, 109]]
        b = [[28, 95], [129, 95]]
        a = [[40, 0], [255, 0]]
        expected = RGB((xs, ys, r, g, b, a), datatype=['grid'], vdims=RGB.vdims + [Dimension('A', range=(0, 1))])
        self.assertEqual(shaded, expected)

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

    def test_shade_dt_xaxis_constant_yaxis(self):
        df = pd.DataFrame({'y': np.ones(100)}, index=pd.date_range('1980-01-01', periods=100, freq='1min'))
        rgb = shade(rasterize(Curve(df), dynamic=False, width=3))
        xs = np.array(['1980-01-01T00:16:30.000000', '1980-01-01T00:49:30.000000', '1980-01-01T01:22:30.000000'], dtype='datetime64[us]')
        ys = np.array([])
        bounds = (np.datetime64('1980-01-01T00:00:00.000000'), 1.0, np.datetime64('1980-01-01T01:39:00.000000'), 1.0)
        expected = RGB((xs, ys, np.empty((0, 3, 4))), ['index', 'y'], xdensity=1, ydensity=1, bounds=bounds)
        self.assertEqual(rgb, expected)
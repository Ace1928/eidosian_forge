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
def test_rasterize_quadmesh(self):
    qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
    img = rasterize(qmesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
    image = Image(np.array([[2, 3, 3], [2, 3, 3], [0, 1, 1]]), bounds=(-0.5, -0.5, 1.5, 1.5))
    self.assertEqual(img, image)
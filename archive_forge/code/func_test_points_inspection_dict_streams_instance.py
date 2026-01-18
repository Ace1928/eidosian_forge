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
def test_points_inspection_dict_streams_instance(self):
    Tap.x, Tap.y = (0.2, 0.3)
    inspector = inspect_points.instance(max_indicators=3, dynamic=True, pixels=1, streams=dict(x=Tap.param.x, y=Tap.param.y))
    points = inspector(self.pntsimg)
    self.assertEqual(len(points.streams), 1)
    self.assertEqual(isinstance(points.streams[0], Tap), True)
    self.assertEqual(points.streams[0].x, 0.2)
    self.assertEqual(points.streams[0].y, 0.3)
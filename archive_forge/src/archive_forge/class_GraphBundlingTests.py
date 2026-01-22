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
class GraphBundlingTests(ComparisonTestCase):

    def setUp(self):
        if ds_version <= Version('0.7.0'):
            raise SkipTest('Regridding operations require datashader>=0.7.0')
        self.source = np.arange(8)
        self.target = np.zeros(8)
        self.graph = Graph(((self.source, self.target),))

    def test_directly_connect_paths(self):
        direct = directly_connect_edges(self.graph)._split_edgepaths
        self.assertEqual(direct, self.graph.edgepaths)
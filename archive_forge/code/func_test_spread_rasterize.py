from unittest import TestCase, SkipTest
import sys
from parameterized import parameterized
import numpy as np
import pandas as pd
from holoviews.core import GridMatrix, NdOverlay
from holoviews.element import (
from hvplot import scatter_matrix
@parameterized.expand([('spread',), ('dynspread',)])
def test_spread_rasterize(self, operation):
    sm = scatter_matrix(self.df, rasterize=True, **{operation: True})
    dm = sm['a', 'b']
    dm[()]
    self.assertEqual(len(dm.last.pipeline.operations), 4)
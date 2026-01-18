from unittest import TestCase, SkipTest
import sys
from parameterized import parameterized
import numpy as np
import pandas as pd
from holoviews.core import GridMatrix, NdOverlay
from holoviews.element import (
from hvplot import scatter_matrix
@parameterized.expand([('rasterize',), ('datashade',)])
def test_rasterization(self, operation):
    sm = scatter_matrix(self.df, **{operation: True})
    dm = sm['a', 'b']
    self.assertEqual(dm.callback.operation.name, operation)
    dm[()]
    self.assertEqual(len(dm.last.pipeline.operations), 3)
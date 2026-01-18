import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_operation_holomap(self):
    hmap = HoloMap({1: Image(np.random.rand(10, 10))})
    op_hmap = operation(hmap, op=lambda x, k: x.clone(x.data * 2))
    self.assertEqual(op_hmap.last, hmap.last.clone(hmap.last.data * 2, group='Operation'))
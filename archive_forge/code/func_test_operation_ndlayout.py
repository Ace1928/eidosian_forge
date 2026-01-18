import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_operation_ndlayout(self):
    ndlayout = NdLayout({i: Image(np.random.rand(10, 10)) for i in range(10)})
    op_ndlayout = operation(ndlayout, op=lambda x, k: x.clone(x.data * 2))
    doubled = ndlayout.clone({k: v.clone(v.data * 2, group='Operation') for k, v in ndlayout.items()})
    self.assertEqual(op_ndlayout, doubled)
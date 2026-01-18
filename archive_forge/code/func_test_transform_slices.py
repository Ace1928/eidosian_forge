from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
def test_transform_slices(self):
    names = self.data.index.names
    transformed_slices = self.grouping.transform_slices(self.data.values, lambda x, idx: x.mean(0), level=0)
    expected = self.data.reset_index().groupby(names[0])[self.data.columns].mean()
    np.testing.assert_allclose(transformed_slices, expected.values, rtol=1e-12, atol=1e-25)
    if len(names) > 1:
        transformed_slices = self.grouping.transform_slices(self.data.values, lambda x, idx: x.mean(0), level=1)
        expected = self.data.reset_index().groupby(names[1])[self.data.columns].mean()
        np.testing.assert_allclose(transformed_slices, expected.values, rtol=1e-12, atol=1e-25)
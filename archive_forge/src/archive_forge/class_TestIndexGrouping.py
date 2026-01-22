from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.tools.grouputils import (dummy_sparse, Grouping, Group,
from statsmodels.datasets import grunfeld, anes96
class TestIndexGrouping(CheckGrouping):

    @classmethod
    def setup_class(cls):
        grun_data = grunfeld.load_pandas().data
        index_data = grun_data.set_index(['firm'])
        index_group = index_data.index
        cls.grouping = Grouping(index_group)
        cls.data = index_data
        cls.expected_counts = [20] * 11
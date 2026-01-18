import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.filterwarnings('ignore:Attempting to set:UserWarning')
def test_parallel_coordinates_with_sorted_labels(self):
    """For #15908"""
    from pandas.plotting import parallel_coordinates
    df = DataFrame({'feat': list(range(30)), 'class': [2 for _ in range(10)] + [3 for _ in range(10)] + [1 for _ in range(10)]})
    ax = parallel_coordinates(df, 'class', sort_labels=True)
    polylines, labels = ax.get_legend_handles_labels()
    color_label_tuples = zip([polyline.get_color() for polyline in polylines], labels)
    ordered_color_label_tuples = sorted(color_label_tuples, key=lambda x: x[1])
    prev_next_tupels = zip(list(ordered_color_label_tuples[0:-1]), list(ordered_color_label_tuples[1:]))
    for prev, nxt in prev_next_tupels:
        assert prev[1] < nxt[1] and prev[0] < nxt[0]
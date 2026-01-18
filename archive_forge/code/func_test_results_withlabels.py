from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
def test_results_withlabels(self):
    labels = ['Test1', 2, 'Aardvark', 4]
    results = cbook.boxplot_stats(self.data, labels=labels)
    for lab, res in zip(labels, results):
        assert res['label'] == lab
    results = cbook.boxplot_stats(self.data)
    for res in results:
        assert 'label' not in res
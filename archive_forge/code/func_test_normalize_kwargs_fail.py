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
@pytest.mark.parametrize('inp, kwargs_to_norm', fail_mapping)
def test_normalize_kwargs_fail(inp, kwargs_to_norm):
    with pytest.raises(TypeError), _api.suppress_matplotlib_deprecation_warning():
        cbook.normalize_kwargs(inp, **kwargs_to_norm)
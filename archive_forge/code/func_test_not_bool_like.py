from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_not_bool_like():
    with pytest.raises(TypeError):
        bool_like(np.array([True, True]), boolean)
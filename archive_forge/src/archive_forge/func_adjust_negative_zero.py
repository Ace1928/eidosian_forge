from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
def adjust_negative_zero(zero, expected):
    """
    Helper to adjust the expected result if we are dividing by -0.0
    as opposed to 0.0
    """
    if np.signbit(np.array(zero)).any():
        assert np.signbit(np.array(zero)).all()
        expected *= -1
    return expected
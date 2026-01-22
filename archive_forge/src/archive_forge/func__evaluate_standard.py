from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._config import get_option
from pandas.util._exceptions import find_stack_level
from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
def _evaluate_standard(op, op_str, a, b):
    """
    Standard evaluation.
    """
    if _TEST_MODE:
        _store_test_result(False)
    return op(a, b)
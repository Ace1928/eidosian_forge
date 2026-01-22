from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._config import get_option
from pandas.util._exceptions import find_stack_level
from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
def _evaluate_numexpr(op, op_str, a, b):
    result = None
    if _can_use_numexpr(op, op_str, a, b, 'evaluate'):
        is_reversed = op.__name__.strip('_').startswith('r')
        if is_reversed:
            a, b = (b, a)
        a_value = a
        b_value = b
        try:
            result = ne.evaluate(f'a_value {op_str} b_value', local_dict={'a_value': a_value, 'b_value': b_value}, casting='safe')
        except TypeError:
            pass
        except NotImplementedError:
            if _bool_arith_fallback(op_str, a, b):
                pass
            else:
                raise
        if is_reversed:
            a, b = (b, a)
    if _TEST_MODE:
        _store_test_result(result is not None)
    if result is None:
        result = _evaluate_standard(op, op_str, a, b)
    return result
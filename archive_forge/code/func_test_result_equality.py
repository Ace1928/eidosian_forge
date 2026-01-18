import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_result_equality():
    et = cirq.testing.EqualsTester()
    et.add_equality_group(cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'a': np.array([[0]] * 5)}), cirq.ResultDict(params=cirq.ParamResolver({}), records={'a': np.array([[[0]]] * 5)}))
    et.add_equality_group(cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'a': np.array([[0]] * 6)}))
    et.add_equality_group(cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'a': np.array([[1]] * 5)}))
    et.add_equality_group(cirq.ResultDict(params=cirq.ParamResolver({}), records={'a': np.array([[[0], [1]]] * 5)}))
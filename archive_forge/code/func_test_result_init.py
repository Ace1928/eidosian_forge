import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_result_init():
    assert cirq.ResultDict(params=cirq.ParamResolver({}), measurements=None).repetitions == 0
    assert cirq.ResultDict(params=cirq.ParamResolver({}), measurements={}).repetitions == 0
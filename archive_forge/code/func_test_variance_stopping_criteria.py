import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_variance_stopping_criteria():
    stop = cw.VarianceStoppingCriteria(variance_bound=1e-06)
    acc = _MockBitstringAccumulator()
    assert stop.more_repetitions(acc) == 10000
    rs = np.random.RandomState(52)
    acc.consume_results(rs.choice([0, 1], size=(100, 5)).astype(np.uint8))
    assert stop.more_repetitions(acc) == 10000
    acc.consume_results(rs.choice([0, 1], size=(10000, 5)).astype(np.uint8))
    assert stop.more_repetitions(acc) == 0
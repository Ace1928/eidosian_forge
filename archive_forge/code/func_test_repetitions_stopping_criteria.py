import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_repetitions_stopping_criteria():
    stop = cw.RepetitionsStoppingCriteria(total_repetitions=50000)
    acc = _MockBitstringAccumulator()
    todos = [stop.more_repetitions(acc)]
    for _ in range(6):
        acc.consume_results(np.zeros((10000, 5), dtype=np.uint8))
        todos.append(stop.more_repetitions(acc))
    assert todos == [10000] * 5 + [0, 0]
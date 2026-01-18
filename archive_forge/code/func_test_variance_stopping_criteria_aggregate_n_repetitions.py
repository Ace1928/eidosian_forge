import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_variance_stopping_criteria_aggregate_n_repetitions():
    stop = _WildVarianceStoppingCriteria()
    acc1 = _MockBitstringAccumulator()
    acc2 = _MockBitstringAccumulator()
    accumulators = {'FakeMeasSpec1': acc1, 'FakeMeasSpec2': acc2}
    with pytest.warns(UserWarning, match='the largest value will be used: 6.'):
        still_todo, reps = _check_meas_specs_still_todo(meas_specs=sorted(accumulators.keys()), accumulators=accumulators, stopping_criteria=stop)
    assert still_todo == ['FakeMeasSpec1', 'FakeMeasSpec2']
    assert reps == 6
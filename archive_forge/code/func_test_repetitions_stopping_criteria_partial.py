import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_repetitions_stopping_criteria_partial():
    stop = cw.RepetitionsStoppingCriteria(total_repetitions=5000, repetitions_per_chunk=1000000)
    acc = _MockBitstringAccumulator()
    assert stop.more_repetitions(acc) == 5000
import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
class _WildVarianceStoppingCriteria(StoppingCriteria):

    def __init__(self):
        self._state = 0

    def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
        """Ignore everything, request either 5 or 6 repetitions."""
        self._state += 1
        return [5, 6][self._state % 2]
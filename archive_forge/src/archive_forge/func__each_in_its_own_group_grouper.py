import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def _each_in_its_own_group_grouper(settings: Iterable[InitObsSetting]) -> Dict[InitObsSetting, List[InitObsSetting]]:
    return {setting: [setting] for setting in settings}
import abc
import dataclasses
import itertools
import os
import tempfile
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, study, ops, value, protocols
from cirq._doc import document
from cirq.work.observable_grouping import group_settings_greedy, GROUPER_T
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import InitObsSetting, observables_to_settings, _MeasurementSpec
def _pad_setting(max_setting: InitObsSetting, qubits: Sequence['cirq.Qid'], pad_init_state_with=value.KET_ZERO, pad_obs_with: 'cirq.Gate'=ops.Z) -> InitObsSetting:
    """Pad `max_setting`'s `init_state` and `observable` with `pad_xx_with` operations
    (defaults:  |0> and Z) so each max_setting has the same qubits. We need this
    to be the case so we can fill in all the parameters, see `_get_params_for_setting`.
    """
    obs = max_setting.observable
    assert obs.coefficient == 1, 'Only the max_setting should be padded.'
    for qubit in qubits:
        if not qubit in obs:
            obs *= pad_obs_with(qubit)
    init_state = max_setting.init_state
    init_state_original_qubits = init_state.qubits
    for qubit in qubits:
        if not qubit in init_state_original_qubits:
            init_state *= pad_init_state_with(qubit)
    return InitObsSetting(init_state=init_state, observable=obs)
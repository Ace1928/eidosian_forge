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
@dataclasses.dataclass(frozen=True)
class _FlippyMeasSpec:
    """Internally, each MeasurementSpec class is split into two
    _FlippyMeasSpecs to support readout symmetrization.

    Bitstring results are combined, so this should be opaque to the user.
    """
    meas_spec: _MeasurementSpec
    flips: np.ndarray
    qubits: Sequence['cirq.Qid']

    def param_tuples(self, *, needs_init_layer=True):
        yield from _get_params_for_setting(self.meas_spec.max_setting, flips=self.flips, qubits=self.qubits, needs_init_layer=needs_init_layer).items()
        yield from self.meas_spec.circuit_params.items()
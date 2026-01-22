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
def _subdivide_meas_specs(meas_specs: Iterable[_MeasurementSpec], repetitions: int, qubits: Sequence['cirq.Qid'], readout_symmetrization: bool) -> Tuple[List[_FlippyMeasSpec], int]:
    """Split measurement specs into sub-jobs for readout symmetrization

    In readout symmetrization, we first run the "normal" circuit followed
    by running the circuit with flipped measurement.
    One _MeasurementSpec is split into two _FlippyMeasSpecs. These are run
    separately but accumulated according to their shared _MeasurementSpec.
    """
    n_qubits = len(qubits)
    flippy_mspecs = []
    for meas_spec in meas_specs:
        all_normal = np.zeros(n_qubits, dtype=bool)
        flippy_mspecs.append(_FlippyMeasSpec(meas_spec=meas_spec, flips=all_normal, qubits=qubits))
        if readout_symmetrization:
            all_flipped = np.ones(n_qubits, dtype=bool)
            flippy_mspecs.append(_FlippyMeasSpec(meas_spec=meas_spec, flips=all_flipped, qubits=qubits))
    if readout_symmetrization:
        repetitions //= 2
    return (flippy_mspecs, repetitions)
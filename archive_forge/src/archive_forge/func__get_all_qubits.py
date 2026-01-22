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
def _get_all_qubits(circuit: 'cirq.AbstractCircuit', observables: Iterable['cirq.PauliString']) -> List['cirq.Qid']:
    """Helper function for `measure_observables` to get all qubits from a circuit and a
    collection of observables."""
    qubit_set = set()
    for obs in observables:
        qubit_set |= set(obs.qubits)
    qubit_set |= circuit.all_qubits()
    return sorted(qubit_set)
from dataclasses import replace
from functools import partial
from typing import Union, Tuple, Sequence
import concurrent.futures
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import (
from pennylane.ops.qubit.observables import BasisStateProjector
from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .default_qubit import accepted_sample_measurement
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
def _measure_expectation(self, meas, tableau_simulator, **kwargs):
    """Measure the expectation value with respect to the state of simulator device."""
    meas_obs = meas.obs
    if isinstance(meas_obs, BasisStateProjector):
        kwargs['prob_states'] = qml.math.array([meas_obs.data[0]])
        return self._measure_probability(qml.probs(wires=meas_obs.wires), tableau_simulator, **kwargs).squeeze()
    coeffs, paulis = _pl_obs_to_linear_comb(meas_obs)
    expecs = qml.math.zeros_like(coeffs)
    for idx, (pauli, wire) in enumerate(paulis):
        pauli_term = ['I'] * max(np.max(list(wire)) + 1, tableau_simulator.num_qubits)
        for op, wr in zip(pauli, wire):
            pauli_term[wr] = op
        stim_pauli = stim.PauliString(''.join(pauli_term))
        expecs[idx] = tableau_simulator.peek_observable_expectation(stim_pauli)
    return qml.math.dot(coeffs, expecs)
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
def _measure_state(self, _, tableau_simulator, **kwargs):
    """Measure the state of the simualtor device."""
    wires = kwargs.get('circuit').wires
    global_phase = kwargs.get('global_phase', qml.GlobalPhase(0.0))
    if self._tableau:
        tableau = tableau_simulator.current_inverse_tableau().inverse()
        x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()
        pl_tableau = np.vstack((np.hstack((x2x, x2z, x_signs.reshape(-1, 1))), np.hstack((z2x, z2z, z_signs.reshape(-1, 1))))).astype(int)
        if pl_tableau.shape == (0, 1) and len(wires):
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return pl_tableau
    state = qml.math.array(tableau_simulator.state_vector(endian='big'))
    if state.shape == (1,) and len(wires):
        state = qml.math.zeros(2 ** len(wires), dtype=complex)
        state[0] = 1.0 + 0j
    return state * qml.matrix(global_phase)[0][0]
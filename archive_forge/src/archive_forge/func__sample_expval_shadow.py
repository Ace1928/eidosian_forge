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
def _sample_expval_shadow(self, meas, stim_circuit, shots, seed):
    """Measures expectation value of a Pauli observable using
        classical shadows from the state of simulator device."""
    bits, recipes = self._sample_classical_shadow(meas, stim_circuit, shots, seed)
    wires_map = list(range(stim_circuit.num_qubits))
    shadow = qml.shadows.ClassicalShadow(bits, recipes, wire_map=wires_map)
    return shadow.expval(meas.H, meas.k)
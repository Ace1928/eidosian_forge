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
def _measure_vn_entropy(self, meas, tableau_simulator, **kwargs):
    """Measure the Von Neumann entropy with respect to the state of simulator device."""
    wires = kwargs.get('circuit').wires
    tableau = tableau_simulator.current_inverse_tableau().inverse()
    z_stabs = qml.math.array([tableau.z_output(wire) for wire in range(len(wires))])
    return self._measure_stabilizer_entropy(z_stabs, list(meas.wires), meas.log_base)
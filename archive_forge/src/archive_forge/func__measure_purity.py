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
def _measure_purity(self, meas, tableau_simulator, **kwargs):
    """Measure the purity of the state of simulator device.

        Computes the state purity using the monotonically decreasing second-order Rényi entropy
        form given in `Sci Rep 13, 4601 (2023) <https://www.nature.com/articles/s41598-023-31273-9>`_.
        We utilize the fact that Rényi entropies are equal for all Rényi indices ``n`` for the
        stabilizer states.

        Args:
            stabilizer (TensorLike): stabilizer set for the system
            wires (Iterable): wires describing the subsystem
            log_base (int): base for the logarithm.

        Returns:
            (float): entanglement entropy of the subsystem
        """
    wires = kwargs.get('circuit').wires
    if wires == meas.wires:
        return qml.math.array(1.0)
    tableau = tableau_simulator.current_inverse_tableau().inverse()
    z_stabs = qml.math.array([tableau.z_output(wire) for wire in range(len(wires))])
    return 2 ** (-self._measure_stabilizer_entropy(z_stabs, list(meas.wires), log_base=2))
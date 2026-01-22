from __future__ import annotations
from collections.abc import Sequence
import typing
import numpy as np
from qiskit.circuit import QuantumCircuit, ControlFlowOp, Measure
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from ..utils import init_observable
def _cross_validate_circuits_parameter_values(circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...]) -> None:
    if len(circuits) != len(parameter_values):
        raise ValueError(f'The number of circuits ({len(circuits)}) does not match the number of parameter value sets ({len(parameter_values)}).')
    for i, (circuit, vector) in enumerate(zip(circuits, parameter_values)):
        if len(vector) != circuit.num_parameters:
            raise ValueError(f'The number of values ({len(vector)}) does not match the number of parameters ({circuit.num_parameters}) for the {i}-th circuit.')
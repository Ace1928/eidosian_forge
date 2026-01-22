from __future__ import annotations
from collections.abc import Sequence
import typing
import numpy as np
from qiskit.circuit import QuantumCircuit, ControlFlowOp, Measure
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from ..utils import init_observable
def _cross_validate_circuits_observables(circuits: tuple[QuantumCircuit, ...], observables: tuple[BaseOperator | PauliSumOp, ...]) -> None:
    if len(circuits) != len(observables):
        raise ValueError(f'The number of circuits ({len(circuits)}) does not match the number of observables ({len(observables)}).')
    for i, (circuit, observable) in enumerate(zip(circuits, observables)):
        if circuit.num_qubits != observable.num_qubits:
            raise ValueError(f'The number of qubits of the {i}-th circuit ({circuit.num_qubits}) does not match the number of qubits of the {i}-th observable ({observable.num_qubits}).')
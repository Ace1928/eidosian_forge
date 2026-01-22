from __future__ import annotations
from collections.abc import Sequence
from itertools import accumulate
import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1, BackendV2, Options
from qiskit.quantum_info import Pauli, PauliList
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts, Result
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
from .base import BaseEstimator, EstimatorResult
from .primitive_job import PrimitiveJob
from .utils import _circuit_key, _observable_key, init_observable
@staticmethod
def _measurement_circuit(num_qubits: int, pauli: Pauli):
    qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
    if not np.any(qubit_indices):
        qubit_indices = [0]
    meas_circuit = QuantumCircuit(QuantumRegister(num_qubits, 'q'), ClassicalRegister(len(qubit_indices), f'__c_{pauli}'))
    for clbit, i in enumerate(qubit_indices):
        if pauli.x[i]:
            if pauli.z[i]:
                meas_circuit.sdg(i)
            meas_circuit.h(i)
        meas_circuit.measure(i, clbit)
    return (meas_circuit, qubit_indices)
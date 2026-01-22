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
def _passmanager_for_measurement_circuits(layout, backend) -> PassManager:
    passmanager = PassManager([SetLayout(layout)])
    if isinstance(backend, BackendV2):
        opt1q = Optimize1qGatesDecomposition(target=backend.target)
    else:
        opt1q = Optimize1qGatesDecomposition(basis=backend.configuration().basis_gates)
    passmanager.append(opt1q)
    if isinstance(backend, BackendV2) and isinstance(backend.coupling_map, CouplingMap):
        coupling_map = backend.coupling_map
        passmanager.append(FullAncillaAllocation(coupling_map))
        passmanager.append(EnlargeWithAncilla())
    elif isinstance(backend, BackendV1) and backend.configuration().coupling_map is not None:
        coupling_map = CouplingMap(backend.configuration().coupling_map)
        passmanager.append(FullAncillaAllocation(coupling_map))
        passmanager.append(EnlargeWithAncilla())
    passmanager.append(ApplyLayout())
    return passmanager
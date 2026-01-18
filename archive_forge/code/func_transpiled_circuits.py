from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.options import Options
from qiskit.result import QuasiDistribution, Result
from qiskit.transpiler.passmanager import PassManager
from .backend_estimator import _prepare_counts, _run_circuits
from .base import BaseSampler, SamplerResult
from .primitive_job import PrimitiveJob
from .utils import _circuit_key
@property
def transpiled_circuits(self) -> list[QuantumCircuit]:
    """
        Transpiled quantum circuits.
        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
    if self._skip_transpilation:
        self._transpiled_circuits = list(self._circuits)
    elif len(self._transpiled_circuits) < len(self._circuits):
        self._transpile()
    return self._transpiled_circuits
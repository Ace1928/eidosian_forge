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
def _transpile(self):
    from qiskit.compiler import transpile
    start = len(self._transpiled_circuits)
    self._transpiled_circuits.extend(transpile(self.preprocessed_circuits[start:], self.backend, **self.transpile_options.__dict__))
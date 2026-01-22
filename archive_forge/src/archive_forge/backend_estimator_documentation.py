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

        Postprocessing for evaluation of expectation value using pauli rotation gates.
        
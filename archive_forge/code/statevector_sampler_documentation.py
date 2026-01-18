from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import warnings
import numpy as np
from numpy.typing import NDArray
from qiskit import ClassicalRegister, QiskitError, QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.quantum_info import Statevector
from .base import BaseSamplerV2
from .base.validation import _has_measure
from .containers import (
from .containers.sampler_pub import SamplerPub
from .containers.bit_array import _min_num_bytes
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction
Return the seed or Generator object for random number generation.
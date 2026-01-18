from __future__ import annotations
from collections.abc import Iterable
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from .base import BaseEstimatorV2
from .containers import EstimatorPubLike, PrimitiveResult, PubResult
from .containers.estimator_pub import EstimatorPub
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction
Return the seed or Generator object for random number generation.
from __future__ import annotations
import itertools
from collections.abc import Iterable
from copy import deepcopy
from typing import Union, cast
import numpy as np
from qiskit.exceptions import QiskitError
from ..operators import Pauli, SparsePauliOp
def _sparse_pauli_op_is_zero(op: SparsePauliOp) -> bool:
    """Returns whether or not this operator represents a zero operation."""
    op = op.simplify()
    return len(op.coeffs) == 1 and op.coeffs[0] == 0
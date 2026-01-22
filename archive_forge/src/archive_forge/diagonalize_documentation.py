from typing import Tuple, Callable, List
import numpy as np
from cirq.linalg import combinators, predicates, tolerance
Finds orthogonal matrices L, R such that L @ matrix @ R is diagonal.

    Args:
        mat: A unitary matrix.
        rtol: Relative numeric error threshold.
        atol: Absolute numeric error threshold.
        check_preconditions: If set, verifies that the input is a unitary matrix
            (to the given tolerances). Defaults to set.

    Returns:
        A triplet (L, d, R) such that L @ mat @ R = diag(d). Both L and R will
        be orthogonal matrices with determinant equal to 1.

    Raises:
        ValueError: Matrices don't meet preconditions (e.g. not real).
    
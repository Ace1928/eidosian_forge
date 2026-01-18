from typing import Any, Iterator, Tuple, Union, TYPE_CHECKING
import numpy as np
import sympy
from cirq import linalg, protocols, value, _compat
from cirq.ops import linear_combinations, pauli_string_phasor
Reconstructs matrix of self from underlying Pauli sum exponentials.

        Raises:
            ValueError: if exponent is parameterized.
        
from typing import List, Callable, TYPE_CHECKING
from scipy.linalg import cossin
import numpy as np
from cirq import ops
from cirq.linalg import decompositions, predicates
def _nth_gray(n: int) -> int:
    return n ^ n >> 1
from typing import Any
import numpy as np
from cirq import devices, protocols, ops, circuits
from cirq.testing import lin_alg_utils
Asserts that cirq.decompose(val) ends at default cirq gateset or a known gate.
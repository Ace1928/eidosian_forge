import warnings
from typing import Any, List, Sequence, Optional
import numpy as np
from cirq import devices, linalg, ops, protocols
from cirq.testing import lin_alg_utils
Uses `val._unitary_` to check `val._qasm_`'s behavior.
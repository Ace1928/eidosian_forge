from typing import AbstractSet, Any, cast, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import sympy
import cirq
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq.ops import eigen_gate, swap_gates
Inits PhasedISwapPowGate.

        Args:
            phase_exponent: The exponent on the Z gates. We conjugate by
                the T gate by default.
            exponent: The exponent on the ISWAP gate, see EigenGate for
                details.
            global_shift: The global_shift on the ISWAP gate, see EigenGate for
                details.
        
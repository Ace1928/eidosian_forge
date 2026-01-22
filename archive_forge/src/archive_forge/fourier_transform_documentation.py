from typing import AbstractSet, Any, Dict, Union
import numpy as np
import sympy
import cirq
from cirq import value, _compat
from cirq.ops import raw_types
Inits QuantumFourierTransformGate.

        Args:
            num_qubits: The number of qubits the gate applies to.
            without_reverse: Whether or not to include the swaps at the end
                of the circuit decomposition that reverse the order of the
                qubits. These are technically necessary in order to perform the
                correct effect, but can almost always be optimized away by just
                performing later operations on different qubits.
        
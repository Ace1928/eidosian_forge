from typing import Dict, Iterable, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np
from cirq import ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _is_swaplike(gate: 'cirq.Gate'):
    if isinstance(gate, ops.SwapPowGate):
        return gate.exponent == 1
    if isinstance(gate, ops.ISwapPowGate):
        return _is_integer((gate.exponent - 1) / 2)
    if isinstance(gate, ops.FSimGate):
        return _is_integer(gate.theta / np.pi - 1 / 2)
    return False
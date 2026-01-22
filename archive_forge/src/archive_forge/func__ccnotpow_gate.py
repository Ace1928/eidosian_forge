import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _ccnotpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> Optional[str]:
    gate = cast(cirq.CCNotPowGate, op.gate)
    if gate._exponent != 1:
        return None
    return formatter.format('CCNOT {0} {1} {2}\n', op.qubits[0], op.qubits[1], op.qubits[2])
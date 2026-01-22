import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _iswappow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.ISwapPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('ISWAP {0} {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format('XY({0}) {1} {2}\n', gate._exponent * np.pi, op.qubits[0], op.qubits[1])
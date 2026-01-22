import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _czpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.CZPowGate, op.gate)
    if gate._exponent == 1:
        return formatter.format('CZ {0} {1}\n', op.qubits[0], op.qubits[1])
    return formatter.format('CPHASE({0}) {1} {2}\n', gate._exponent * np.pi, op.qubits[0], op.qubits[1])
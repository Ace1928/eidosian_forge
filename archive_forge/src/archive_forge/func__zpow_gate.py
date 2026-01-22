import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _zpow_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(cirq.ZPowGate, op.gate)
    if gate._exponent == 1 and gate.global_shift != -0.5:
        return formatter.format('Z {0}\n', op.qubits[0])
    return formatter.format('RZ({0}) {1}\n', gate._exponent * np.pi, op.qubits[0])
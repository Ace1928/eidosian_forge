import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _quiltwoqubit_gate(op: cirq.Operation, formatter: QuilFormatter) -> str:
    gate = cast(QuilOneQubitGate, op.gate)
    return f'DEFGATE USERGATE:\n    {to_quil_complex_format(gate.matrix[0, 0])}, {to_quil_complex_format(gate.matrix[0, 1])}, {to_quil_complex_format(gate.matrix[0, 2])}, {to_quil_complex_format(gate.matrix[0, 3])}\n    {to_quil_complex_format(gate.matrix[1, 0])}, {to_quil_complex_format(gate.matrix[1, 1])}, {to_quil_complex_format(gate.matrix[1, 2])}, {to_quil_complex_format(gate.matrix[1, 3])}\n    {to_quil_complex_format(gate.matrix[2, 0])}, {to_quil_complex_format(gate.matrix[2, 1])}, {to_quil_complex_format(gate.matrix[2, 2])}, {to_quil_complex_format(gate.matrix[2, 3])}\n    {to_quil_complex_format(gate.matrix[3, 0])}, {to_quil_complex_format(gate.matrix[3, 1])}, {to_quil_complex_format(gate.matrix[3, 2])}, {to_quil_complex_format(gate.matrix[3, 3])}\n{formatter.format('USERGATE {0} {1}', op.qubits[0], op.qubits[1])}\n'
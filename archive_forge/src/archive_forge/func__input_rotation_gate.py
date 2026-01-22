from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import cirq
from cirq import ops, linalg, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
def _input_rotation_gate(identifier: str, gate: 'cirq.Gate', exponent_sign: int) -> CellMaker:
    return CellMaker(identifier, gate.num_qubits(), lambda args: InputRotationCell(identifier=identifier, register=None, base_operation=gate.on(args.qubits[0]), exponent_sign=exponent_sign))
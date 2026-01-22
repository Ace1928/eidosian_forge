from __future__ import annotations
from typing import Iterator, Iterable
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from .annotated_operation import AnnotatedOperation, ControlModifier
from .instruction import Instruction
@staticmethod
def _broadcast_2_arguments(qarg0: list, qarg1: list) -> Iterator[tuple[list, list]]:
    if len(qarg0) == len(qarg1):
        for arg0, arg1 in zip(qarg0, qarg1):
            yield ([arg0, arg1], [])
    elif len(qarg0) == 1:
        for arg1 in qarg1:
            yield ([qarg0[0], arg1], [])
    elif len(qarg1) == 1:
        for arg0 in qarg0:
            yield ([arg0, qarg1[0]], [])
    else:
        raise CircuitError(f'Not sure how to combine these two-qubit arguments:\n {qarg0}\n {qarg1}')
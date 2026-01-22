from __future__ import annotations
from typing import Iterator, Iterable
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from .annotated_operation import AnnotatedOperation, ControlModifier
from .instruction import Instruction
@staticmethod
def _broadcast_3_or_more_args(qargs: list) -> Iterator[tuple[list, list]]:
    if all((len(qarg) == len(qargs[0]) for qarg in qargs)):
        for arg in zip(*qargs):
            yield (list(arg), [])
    else:
        raise CircuitError('Not sure how to combine these qubit arguments:\n %s\n' % qargs)
import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
def _decompose_with_context_(self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext]=None) -> cirq.OP_TREE:
    qubit_regs = split_qubits(self.signature, qubits)
    if context is None:
        context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    return self.decompose_from_registers(context=context, **qubit_regs)
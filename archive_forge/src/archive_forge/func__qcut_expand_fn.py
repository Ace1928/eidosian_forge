import copy
from itertools import product
from typing import Callable, List, Sequence, Tuple, Union
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import expval
from pennylane.measurements import ExpectationMP, MeasurementProcess, SampleMP
from pennylane.operation import Operator, Tensor
from pennylane.ops.meta import WireCut
from pennylane.pauli import string_to_pauli_word
from pennylane.queuing import WrappedObj
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.wires import Wires
from .utils import MeasureNode, PrepareNode
def _qcut_expand_fn(tape: QuantumTape, max_depth: int=1, auto_cutter: Union[bool, Callable]=False):
    """Expansion function for circuit cutting.

    Expands operations until reaching a depth that includes :class:`~.WireCut` operations.
    """
    for op in tape.operations:
        if isinstance(op, WireCut):
            return tape
    if max_depth > 0:
        return _qcut_expand_fn(tape.expand(), max_depth=max_depth - 1, auto_cutter=auto_cutter)
    if not (auto_cutter is True or callable(auto_cutter)):
        raise ValueError('No WireCut operations found in the circuit. Consider increasing the max_depth value if operations or nested tapes contain WireCut operations.')
    return tape
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
def _create_prep_list():
    """
    Creates a predetermined list for converting PrepareNodes to an associated Operation for use
    within the expand_fragment_tape function.
    """

    def _prep_zero(wire):
        return [qml.Identity(wire)]

    def _prep_one(wire):
        return [qml.X(wire)]

    def _prep_plus(wire):
        return [qml.Hadamard(wire)]

    def _prep_iplus(wire):
        return [qml.Hadamard(wire), qml.S(wires=wire)]
    return [_prep_zero, _prep_one, _prep_plus, _prep_iplus]
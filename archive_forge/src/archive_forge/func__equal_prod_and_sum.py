from collections.abc import Iterable
from functools import singledispatch
from typing import Union
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.measurements.classical_shadow import ShadowExpvalMP
from pennylane.measurements.mid_measure import MidMeasureMP, MeasurementValue
from pennylane.measurements.mutual_info import MutualInfoMP
from pennylane.measurements.vn_entropy import VnEntropyMP
from pennylane.measurements.counts import CountsMP
from pennylane.pulse.parametrized_evolution import ParametrizedEvolution
from pennylane.operation import Observable, Operator, Tensor
from pennylane.ops import Hamiltonian, Controlled, Pow, Adjoint, Exp, SProd, CompositeOp
from pennylane.templates.subroutines import ControlledSequence
from pennylane.tape import QuantumTape
@_equal.register
def _equal_prod_and_sum(op1: CompositeOp, op2: CompositeOp, **kwargs):
    """Determine whether two Prod or Sum objects are equal"""
    if op1.pauli_rep is not None and op1.pauli_rep == op2.pauli_rep:
        return True
    if len(op1.operands) != len(op2.operands):
        return False
    sorted_ops1 = op1._sort(op1.operands)
    sorted_ops2 = op2._sort(op2.operands)
    return all((equal(o1, o2, **kwargs) for o1, o2 in zip(sorted_ops1, sorted_ops2)))
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
def _equal_shadow_measurements(op1: ShadowExpvalMP, op2: ShadowExpvalMP, **_):
    """Determine whether two ShadowExpvalMP objects are equal"""
    wires_match = op1.wires == op2.wires
    if isinstance(op1.H, Operator) and isinstance(op2.H, Operator):
        H_match = equal(op1.H, op2.H)
    elif isinstance(op1.H, Iterable) and isinstance(op2.H, Iterable):
        H_match = all((equal(o1, o2) for o1, o2 in zip(op1.H, op2.H)))
    else:
        return False
    k_match = op1.k == op2.k
    return wires_match and H_match and k_match
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
def _equal_measurements(op1: MeasurementProcess, op2: MeasurementProcess, check_interface=True, check_trainability=True, rtol=1e-05, atol=1e-09):
    """Determine whether two MeasurementProcess objects are equal"""
    if op1.obs is not None and op2.obs is not None:
        return equal(op1.obs, op2.obs, check_interface=check_interface, check_trainability=check_trainability, rtol=rtol, atol=atol)
    if op1.mv is not None and op2.mv is not None:
        if isinstance(op1.mv, MeasurementValue) and isinstance(op2.mv, MeasurementValue):
            return op1.mv.measurements == op2.mv.measurements
        if isinstance(op1.mv, Iterable) and isinstance(op2.mv, Iterable):
            if len(op1.mv) == len(op2.mv):
                return all((mv1.measurements == mv2.measurements for mv1, mv2 in zip(op1.mv, op2.mv)))
        return False
    if op1.wires != op2.wires:
        return False
    if op1.obs is None and op2.obs is None:
        if op1.eigvals() is not None and op2.eigvals() is not None:
            return qml.math.allclose(op1.eigvals(), op2.eigvals(), rtol=rtol, atol=atol)
        return op1.eigvals() is None and op2.eigvals() is None
    return False
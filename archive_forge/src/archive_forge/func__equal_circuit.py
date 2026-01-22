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
def _equal_circuit(op1: qml.tape.QuantumScript, op2: qml.tape.QuantumScript, check_interface=True, check_trainability=True, rtol=1e-05, atol=1e-09):
    if len(op1.operations) != len(op2.operations):
        return False
    for comparands in zip(op1.operations, op2.operations):
        if not qml.equal(comparands[0], comparands[1], check_interface=check_interface, check_trainability=check_trainability, rtol=rtol, atol=atol):
            return False
    if len(op1.measurements) != len(op2.measurements):
        return False
    for comparands in zip(op1.measurements, op2.measurements):
        if not qml.equal(comparands[0], comparands[1], check_interface=check_interface, check_trainability=check_trainability, rtol=rtol, atol=atol):
            return False
    if op1.shots != op2.shots:
        return False
    if op1.trainable_params != op2.trainable_params:
        return False
    return True
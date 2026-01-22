from __future__ import annotations
import contextlib
from typing import Union, Iterable, Any, Tuple, Optional, List, Literal, TYPE_CHECKING
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from .builder import InstructionPlaceholder, InstructionResources, ControlFlowBuilderBlock
from .control_flow import ControlFlowOp
from ._builder_utils import unify_circuit_resources, partition_registers, node_resources
class CaseBuilder:
    """A child context manager for building up the ``case`` blocks of ``switch`` statements onto
    circuits in a natural order, without having to construct the case bodies first.

    This context should never need to be created manually by a user; it is the return value of the
    :class:`.SwitchContext` context manager, which in turn should only be created by suitable
    :meth:`.QuantumCircuit.switch_case` calls.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """
    DEFAULT = CASE_DEFAULT
    'Convenient re-exposure of the :data:`.CASE_DEFAULT` constant.'

    def __init__(self, parent: SwitchContext):
        self.switch = parent
        self.entered = False

    @contextlib.contextmanager
    def __call__(self, *values):
        if self.entered:
            raise CircuitError('Cannot enter more than one case at once. If you want multiple labels to point to the same block, pass them all to a single case context, such as `with case(1, 2, 3):`.')
        if self.switch.complete:
            raise CircuitError('Cannot add a new case to a completed switch statement.')
        if not all((value is CASE_DEFAULT or isinstance(value, int) for value in values)):
            raise CircuitError('Case values must be integers or `CASE_DEFAULT`')
        seen = set()
        for value in values:
            if self.switch.label_in_use(value) or value in seen:
                raise CircuitError(f"duplicate case label: '{value}'")
            seen.add(value)
        self.switch.circuit._push_scope(clbits=self.switch.target_clbits, registers=self.switch.target_cregs, allow_jumps=self.switch.in_loop)
        try:
            self.entered = True
            yield
        finally:
            self.entered = False
            block = self.switch.circuit._pop_scope()
        self.switch.add_case(values, block)
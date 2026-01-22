from __future__ import annotations
import contextlib
from typing import Union, Iterable, Any, Tuple, Optional, List, Literal, TYPE_CHECKING
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from .builder import InstructionPlaceholder, InstructionResources, ControlFlowBuilderBlock
from .control_flow import ControlFlowOp
from ._builder_utils import unify_circuit_resources, partition_registers, node_resources
class SwitchContext:
    """A context manager for building up ``switch`` statements onto circuits in a natural order,
    without having to construct the case bodies first.

    The return value of this context manager can be used within the created context to build up the
    individual ``case`` statements.  No other instructions should be appended to the circuit during
    the `switch` context.

    This context should almost invariably be created by a :meth:`.QuantumCircuit.switch_case` call,
    and the resulting instance is a "friend" of the calling circuit.  The context will manipulate
    the circuit's defined scopes when it is entered (by pushing a new scope onto the stack) and
    exited (by popping its scope, building it, and appending the resulting :obj:`.SwitchCaseOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(self, circuit: QuantumCircuit, target: Clbit | ClassicalRegister | expr.Expr, *, in_loop: bool, label: Optional[str]=None):
        self.circuit = circuit
        self._target = target
        if isinstance(target, Clbit):
            self.target_clbits: tuple[Clbit, ...] = (target,)
            self.target_cregs: tuple[ClassicalRegister, ...] = ()
        elif isinstance(target, ClassicalRegister):
            self.target_clbits = tuple(target)
            self.target_cregs = (target,)
        else:
            resources = node_resources(target)
            self.target_clbits = resources.clbits
            self.target_cregs = resources.cregs
        self.in_loop = in_loop
        self.complete = False
        self._op_label = label
        self._cases: List[Tuple[Tuple[Any, ...], ControlFlowBuilderBlock]] = []
        self._label_set = set()

    def label_in_use(self, label):
        """Return whether a case label is already accounted for in the switch statement."""
        return label in self._label_set

    def add_case(self, labels: Tuple[Union[int, Literal[CASE_DEFAULT]], ...], block: ControlFlowBuilderBlock):
        """Add a sequence of conditions and the single block that should be run if they are
        triggered to the context.  The labels are assumed to have already been validated using
        :meth:`label_in_use`."""
        self._label_set.update(labels)
        self._cases.append((labels, block))

    def __enter__(self):
        self.circuit._push_scope(forbidden_message='Cannot have instructions outside a case')
        return CaseBuilder(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.complete = True
        self.circuit._pop_scope()
        if exc_type is not None:
            return False
        placeholder = SwitchCasePlaceholder(self._target, self._cases, label=self._op_label)
        initial_resources = placeholder.placeholder_resources()
        if self.in_loop:
            self.circuit.append(placeholder, initial_resources.qubits, initial_resources.clbits)
        else:
            operation, resources = placeholder.concrete_instruction(set(initial_resources.qubits), set(initial_resources.clbits))
            self.circuit.append(operation, resources.qubits, resources.clbits)
        return False
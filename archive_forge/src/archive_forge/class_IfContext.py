from __future__ import annotations
from typing import Optional, Union, Iterable, TYPE_CHECKING
import itertools
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr
from qiskit.circuit.instructionset import InstructionSet
from qiskit.circuit.exceptions import CircuitError
from .builder import ControlFlowBuilderBlock, InstructionPlaceholder, InstructionResources
from .control_flow import ControlFlowOp
from ._builder_utils import (
class IfContext:
    """A context manager for building up ``if`` statements onto circuits in a natural order, without
    having to construct the statement body first.

    The return value of this context manager can be used immediately following the block to create
    an attached ``else`` statement.

    This context should almost invariably be created by a :meth:`.QuantumCircuit.if_test` call, and
    the resulting instance is a "friend" of the calling circuit.  The context will manipulate the
    circuit's defined scopes when it is entered (by pushing a new scope onto the stack) and exited
    (by popping its scope, building it, and appending the resulting :obj:`.IfElseOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """
    __slots__ = ('_appended_instructions', '_circuit', '_condition', '_in_loop', '_label')

    def __init__(self, circuit: QuantumCircuit, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, *, in_loop: bool, label: str | None=None):
        self._circuit = circuit
        self._condition = validate_condition(condition)
        self._label = label
        self._appended_instructions = None
        self._in_loop = in_loop

    @property
    def circuit(self) -> QuantumCircuit:
        """Get the circuit that this context manager is attached to."""
        return self._circuit

    @property
    def condition(self) -> tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr:
        """Get the expression that this statement is conditioned on."""
        return self._condition

    @property
    def appended_instructions(self) -> Union[InstructionSet, None]:
        """Get the instruction set that was created when this block finished.  If the block has not
        yet finished, then this will be ``None``."""
        return self._appended_instructions

    @property
    def in_loop(self) -> bool:
        """Whether this context manager is enclosed within a loop."""
        return self._in_loop

    def __enter__(self):
        resources = condition_resources(self._condition)
        self._circuit._push_scope(clbits=resources.clbits, registers=resources.cregs, allow_jumps=self._in_loop)
        return ElseContext(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._circuit._pop_scope()
            return False
        true_block = self._circuit._pop_scope()
        if self._in_loop:
            operation = IfElsePlaceholder(self._condition, true_block, label=self._label)
            resources = operation.placeholder_resources()
            self._appended_instructions = self._circuit.append(operation, resources.qubits, resources.clbits)
        else:
            true_body = true_block.build(true_block.qubits(), true_block.clbits())
            self._appended_instructions = self._circuit.append(IfElseOp(self._condition, true_body=true_body, false_body=None, label=self._label), tuple(true_body.qubits), tuple(true_body.clbits))
        return False
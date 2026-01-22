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
class IfElseOp(ControlFlowOp):
    """A circuit operation which executes a program (``true_body``) if a
    provided condition (``condition``) evaluates to true, and
    optionally evaluates another program (``false_body``) otherwise.

    Parameters:
        condition: A condition to be evaluated at circuit runtime which,
            if true, will trigger the evaluation of ``true_body``. Can be
            specified as either a tuple of a ``ClassicalRegister`` to be
            tested for equality with a given ``int``, or as a tuple of a
            ``Clbit`` to be compared to either a ``bool`` or an ``int``.
        true_body: A program to be executed if ``condition`` evaluates
            to true.
        false_body: A optional program to be executed if ``condition``
            evaluates to false.
        label: An optional label for identifying the instruction.

    If provided, ``false_body`` must be of the same ``num_qubits`` and
    ``num_clbits`` as ``true_body``.

    The classical bits used in ``condition`` must be a subset of those attached
    to the circuit on which this ``IfElseOp`` will be appended.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1          ├
             │  if_else  │
        q_2: ┤2          ├
             │           │
        c_0: ╡0          ╞
             └───────────┘

    """

    def __init__(self, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, true_body: QuantumCircuit, false_body: QuantumCircuit | None=None, label: str | None=None):
        from qiskit.circuit import QuantumCircuit
        if not isinstance(true_body, QuantumCircuit):
            raise CircuitError(f'IfElseOp expects a true_body parameter of type QuantumCircuit, but received {type(true_body)}.')
        num_qubits = true_body.num_qubits
        num_clbits = true_body.num_clbits
        super().__init__('if_else', num_qubits, num_clbits, [true_body, false_body], label=label)
        self.condition = validate_condition(condition)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):
        from qiskit.circuit import QuantumCircuit
        true_body, false_body = parameters
        if not isinstance(true_body, QuantumCircuit):
            raise CircuitError(f'IfElseOp expects a true_body parameter of type QuantumCircuit, but received {type(true_body)}.')
        if true_body.num_qubits != self.num_qubits or true_body.num_clbits != self.num_clbits:
            raise CircuitError(f'Attempted to assign a true_body parameter with a num_qubits or num_clbits different than that of the IfElseOp. IfElseOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} Supplied body num_qubits/clbits: {true_body.num_qubits}/{true_body.num_clbits}.')
        if false_body is not None:
            if not isinstance(false_body, QuantumCircuit):
                raise CircuitError(f'IfElseOp expects a false_body parameter of type QuantumCircuit, but received {type(false_body)}.')
            if false_body.num_qubits != self.num_qubits or false_body.num_clbits != self.num_clbits:
                raise CircuitError(f'Attempted to assign a false_body parameter with a num_qubits or num_clbits different than that of the IfElseOp. IfElseOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} Supplied body num_qubits/clbits: {false_body.num_qubits}/{false_body.num_clbits}.')
        self._params = [true_body, false_body]

    @property
    def blocks(self):
        if self.params[1] is None:
            return (self.params[0],)
        else:
            return (self.params[0], self.params[1])

    def replace_blocks(self, blocks: Iterable[QuantumCircuit]) -> 'IfElseOp':
        """Replace blocks and return new instruction.

        Args:
            blocks: Iterable of circuits for "if" and "else" condition. If there is no "else"
                circuit it may be set to None or omitted.

        Returns:
            New IfElseOp with replaced blocks.
        """
        true_body, false_body = (ablock for ablock, _ in itertools.zip_longest(blocks, range(2), fillvalue=None))
        return IfElseOp(self.condition, true_body, false_body=false_body, label=self.label)

    def c_if(self, classical, val):
        raise NotImplementedError('IfElseOp cannot be classically controlled through Instruction.c_if. Please nest it in an IfElseOp instead.')
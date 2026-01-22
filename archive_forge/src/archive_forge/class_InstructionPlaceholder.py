from __future__ import annotations
import abc
import itertools
import typing
from typing import Collection, Iterable, FrozenSet, Tuple, Union, Optional, Sequence
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.circuit.classicalregister import Clbit, ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.circuit.register import Register
from ._builder_utils import condition_resources, node_resources
class InstructionPlaceholder(Instruction, abc.ABC):
    """A fake instruction that lies about its number of qubits and clbits.

    These instances are used to temporarily represent control-flow instructions during the builder
    process, when their lengths cannot be known until the end of the block.  This is necessary to
    allow constructs like::

        with qc.for_loop(range(5)):
            qc.h(0)
            qc.measure(0, 0)
            qc.break_loop().c_if(0, 0)

    since ``qc.break_loop()`` needs to return a (mostly) functional
    :obj:`~qiskit.circuit.Instruction` in order for :meth:`.InstructionSet.c_if` to work correctly.

    When appending a placeholder instruction into a circuit scope, you should create the
    placeholder, and then ask it what resources it should be considered as using from the start by
    calling :meth:`.InstructionPlaceholder.placeholder_instructions`.  This set will be a subset of
    the final resources it asks for, but it is used for initialising resources that *must* be
    supplied, such as the bits used in the conditions of placeholder ``if`` statements.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """
    _directive = True

    @abc.abstractmethod
    def concrete_instruction(self, qubits: FrozenSet[Qubit], clbits: FrozenSet[Clbit]) -> Tuple[Instruction, InstructionResources]:
        """Get a concrete, complete instruction that is valid to act over all the given resources.

        The returned resources may not be the full width of the given resources, but will certainly
        be a subset of them; this can occur if (for example) a placeholder ``if`` statement is
        present, but does not itself contain any placeholder instructions.  For resource efficiency,
        the returned :class:`.ControlFlowOp` will not unnecessarily span all resources, but only the
        ones that it needs.

        .. note::

            The caller of this function is responsible for ensuring that the inputs to this function
            are non-strict supersets of the bits returned by :meth:`placeholder_resources`.

        Any condition added in by a call to :obj:`.Instruction.c_if` will be propagated through, but
        set properties like ``duration`` will not; it doesn't make sense for control-flow operations
        to have pulse scheduling on them.

        Args:
            qubits: The qubits the created instruction should be defined across.
            clbits: The clbits the created instruction should be defined across.

        Returns:
            A full version of the relevant control-flow instruction, and the resources that it uses.
            This is a "proper" instruction instance, as if it had been defined with the correct
            number of qubits and clbits from the beginning.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def placeholder_resources(self) -> InstructionResources:
        """Get the qubit and clbit resources that this placeholder instruction should be considered
        as using before construction.

        This will likely not include *all* resources after the block has been built, but using the
        output of this method ensures that all resources will pass through a
        :meth:`.QuantumCircuit.append` call, even if they come from a placeholder, and consequently
        will be tracked by the scope managers.

        Returns:
            A collection of the quantum and classical resources this placeholder instruction will
            certainly use.
        """
        raise NotImplementedError

    def _copy_mutable_properties(self, instruction: Instruction) -> Instruction:
        """Copy mutable properties from ourselves onto a non-placeholder instruction.

        The mutable properties are expected to be things like ``condition``, added onto a
        placeholder by the :meth:`c_if` method.  This mutates ``instruction``, and returns the same
        instance that was passed.  This is mostly intended to make writing concrete versions of
        :meth:`.concrete_instruction` easy.

        The complete list of mutations is:

        * ``condition``, added by :meth:`c_if`.

        Args:
            instruction: the concrete instruction instance to be mutated.

        Returns:
            The same instruction instance that was passed, but mutated to propagate the tracked
            changes to this class.
        """
        instruction.condition = self.condition
        return instruction

    def assemble(self):
        raise CircuitError('Cannot assemble a placeholder instruction.')

    def repeat(self, n):
        raise CircuitError('Cannot repeat a placeholder instruction.')
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
def add_bits(self, bits: Iterable[Union[Qubit, Clbit]]):
    """Add extra bits to this scope that are not associated with any concrete instruction yet.

        This is useful for expanding a scope's resource width when it may contain ``break`` or
        ``continue`` statements, or when its width needs to be expanded to match another scope's
        width (as in the case of :obj:`.IfElseOp`).

        Args:
            bits: The qubits and clbits that should be added to a scope.  It is not an error if
                there are duplicates, either within the iterable or with the bits currently in
                scope.

        Raises:
            TypeError: if the provided bit is of an incorrect type.
        """
    for bit in bits:
        if isinstance(bit, Qubit):
            self.instructions.add_qubit(bit, strict=False)
        elif isinstance(bit, Clbit):
            self.instructions.add_clbit(bit, strict=False)
        else:
            raise TypeError(f"Can only add qubits or classical bits, but received '{bit}'.")
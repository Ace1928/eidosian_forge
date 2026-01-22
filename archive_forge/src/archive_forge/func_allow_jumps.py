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
@property
def allow_jumps(self):
    """Whether this builder scope should allow ``break`` and ``continue`` statements within it.

        This is intended to help give sensible error messages when dangerous behaviour is
        encountered, such as using ``break`` inside an ``if`` context manager that is not within a
        ``for`` manager.  This can only be safe if the user is going to place the resulting
        :obj:`.QuantumCircuit` inside a :obj:`.ForLoopOp` that uses *exactly* the same set of
        resources.  We cannot verify this from within the builder interface (and it is too expensive
        to do when the ``for`` op is made), so we fail safe, and require the user to use the more
        verbose, internal form.
        """
    return self._allow_jumps
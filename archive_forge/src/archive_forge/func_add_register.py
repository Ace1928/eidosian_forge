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
def add_register(self, register: Register):
    """Add a :obj:`.Register` to the set of resources used by this block, ensuring that
        all bits contained within are also accounted for.

        Args:
            register: the register to add to the block.
        """
    if register in self.registers:
        return
    self.registers.add(register)
    self.add_bits(register)
from __future__ import annotations
import copy
from itertools import zip_longest
import math
from typing import List, Type
import numpy
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.qobj.qasm_qobj import QasmQobjInstruction
from qiskit.circuit.parameter import ParameterExpression
from qiskit.circuit.operation import Operation
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier
def to_mutable(self):
    """Return a mutable copy of this gate.

        This method will return a new mutable copy of this gate instance.
        If a singleton instance is being used this will be a new unique
        instance that can be mutated. If the instance is already mutable it
        will be a deepcopy of that instance.
        """
    return self.copy()
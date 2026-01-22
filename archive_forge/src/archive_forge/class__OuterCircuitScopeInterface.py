from __future__ import annotations
import copy
import multiprocessing as mp
import typing
from collections import OrderedDict, defaultdict, namedtuple
from typing import (
import numpy as np
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import is_main_process
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from . import _classical_resource_map
from ._utils import sort_parameters
from .controlflow.builder import CircuitScopeInterface, ControlFlowBuilderBlock
from .controlflow.break_loop import BreakLoopOp, BreakLoopPlaceholder
from .controlflow.continue_loop import ContinueLoopOp, ContinueLoopPlaceholder
from .controlflow.for_loop import ForLoopOp, ForLoopContext
from .controlflow.if_else import IfElseOp, IfContext
from .controlflow.switch_case import SwitchCaseOp, SwitchContext
from .controlflow.while_loop import WhileLoopOp, WhileLoopContext
from .classical import expr
from .parameterexpression import ParameterExpression, ParameterValueType
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterReferences, ParameterTable, ParameterView
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .operation import Operation
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData, CircuitInstruction
from .delay import Delay
class _OuterCircuitScopeInterface(CircuitScopeInterface):
    __slots__ = ('circuit',)

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit

    @property
    def instructions(self):
        return self.circuit._data

    def append(self, instruction):
        return self.circuit._append(instruction)

    def extend(self, data: CircuitData):
        self.circuit._data.extend(data)
        data.foreach_op(self.circuit._track_operation)

    def resolve_classical_resource(self, specifier):
        if isinstance(specifier, Clbit):
            if specifier not in self.circuit._clbit_indices:
                raise CircuitError(f'Clbit {specifier} is not present in this circuit.')
            return specifier
        if isinstance(specifier, ClassicalRegister):
            if specifier not in self.circuit.cregs:
                raise CircuitError(f'Register {specifier} is not present in this circuit.')
            return specifier
        if isinstance(specifier, int):
            try:
                return self.circuit._data.clbits[specifier]
            except IndexError:
                raise CircuitError(f'Classical bit index {specifier} is out-of-range.') from None
        raise CircuitError(f"Unknown classical resource specifier: '{specifier}'.")
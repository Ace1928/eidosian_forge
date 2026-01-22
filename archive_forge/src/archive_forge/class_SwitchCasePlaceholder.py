from __future__ import annotations
import contextlib
from typing import Union, Iterable, Any, Tuple, Optional, List, Literal, TYPE_CHECKING
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from .builder import InstructionPlaceholder, InstructionResources, ControlFlowBuilderBlock
from .control_flow import ControlFlowOp
from ._builder_utils import unify_circuit_resources, partition_registers, node_resources
class SwitchCasePlaceholder(InstructionPlaceholder):
    """A placeholder instruction to use in control-flow context managers, when calculating the
    number of resources this instruction should block is deferred until the construction of the
    outer loop.

    This generally should not be instantiated manually; only :obj:`.SwitchContext` should do it when
    it needs to defer creation of the concrete instruction.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(self, target: Clbit | ClassicalRegister | expr.Expr, cases: List[Tuple[Any, ControlFlowBuilderBlock]], *, label: Optional[str]=None):
        self.__target = target
        self.__cases = cases
        self.__resources = self._calculate_placeholder_resources()
        super().__init__('switch_case', len(self.__resources.qubits), len(self.__resources.clbits), [], label=label)

    def _calculate_placeholder_resources(self):
        qubits = set()
        clbits = set()
        qregs = set()
        cregs = set()
        if isinstance(self.__target, Clbit):
            clbits.add(self.__target)
        elif isinstance(self.__target, ClassicalRegister):
            clbits.update(self.__target)
            cregs.add(self.__target)
        else:
            resources = node_resources(self.__target)
            clbits.update(resources.clbits)
            cregs.update(resources.cregs)
        for _, body in self.__cases:
            qubits |= body.qubits()
            clbits |= body.clbits()
            body_qregs, body_cregs = partition_registers(body.registers)
            qregs |= body_qregs
            cregs |= body_cregs
        return InstructionResources(qubits=tuple(qubits), clbits=tuple(clbits), qregs=tuple(qregs), cregs=tuple(cregs))

    def placeholder_resources(self):
        return self.__resources

    def concrete_instruction(self, qubits, clbits):
        cases = [(labels, unified_body) for (labels, _), unified_body in zip(self.__cases, unify_circuit_resources((body.build(qubits, clbits) for _, body in self.__cases)))]
        if cases:
            resources = InstructionResources(qubits=tuple(cases[0][1].qubits), clbits=tuple(cases[0][1].clbits), qregs=tuple(cases[0][1].qregs), cregs=tuple(cases[0][1].cregs))
        else:
            resources = self.__resources
        return (self._copy_mutable_properties(SwitchCaseOp(self.__target, cases, label=self.label)), resources)
from __future__ import annotations
import abc
from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable
from inspect import signature
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.base_tasks import GenericPass, PassManagerIR
from qiskit.passmanager.compilation_status import PropertySet, RunState, PassManagerState
from .exceptions import TranspilerError
from .layout import TranspileLayout
class TransformationPass(BasePass):
    """A transformation pass: change DAG, not property set."""

    def execute(self, passmanager_ir: PassManagerIR, state: PassManagerState, callback: Callable=None) -> tuple[PassManagerIR, PassManagerState]:
        new_dag, state = super().execute(passmanager_ir=passmanager_ir, state=state, callback=callback)
        if state.workflow_status.previous_run == RunState.SUCCESS:
            if isinstance(new_dag, DAGCircuit):
                new_dag.calibrations = passmanager_ir.calibrations
            else:
                raise TranspilerError(f'Transformation passes should return a transformed dag.The pass {self.__class__.__name__} is returning a {type(new_dag)}')
        return (new_dag, state)

    def update_status(self, state: PassManagerState, run_state: RunState) -> PassManagerState:
        state = super().update_status(state, run_state)
        if run_state == RunState.SUCCESS:
            state.workflow_status.completed_passes.intersection_update(set(self.preserves))
        return state
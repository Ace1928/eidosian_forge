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
@staticmethod
def _freeze_init_parameters(class_, args, kwargs):
    self_guard = object()
    init_signature = signature(class_.__init__)
    bound_signature = init_signature.bind(self_guard, *args, **kwargs)
    arguments = [('class_.__name__', class_.__name__)]
    for name, value in bound_signature.arguments.items():
        if value == self_guard:
            continue
        if isinstance(value, Hashable):
            arguments.append((name, type(value), value))
        else:
            arguments.append((name, type(value), repr(value)))
    return frozenset(arguments)
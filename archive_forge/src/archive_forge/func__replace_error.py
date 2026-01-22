from __future__ import annotations
import inspect
import io
import re
from collections.abc import Iterator, Iterable, Callable
from functools import wraps
from typing import Union, List, Any
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.passmanager import BasePassManager
from qiskit.passmanager.base_tasks import Task
from qiskit.passmanager.flow_controllers import FlowControllerLinear
from qiskit.passmanager.exceptions import PassManagerError
from .basepasses import BasePass
from .exceptions import TranspilerError
from .layout import TranspileLayout
def _replace_error(meth):

    @wraps(meth)
    def wrapper(*meth_args, **meth_kwargs):
        try:
            return meth(*meth_args, **meth_kwargs)
        except PassManagerError as ex:
            raise TranspilerError(ex.message) from ex
    return wrapper
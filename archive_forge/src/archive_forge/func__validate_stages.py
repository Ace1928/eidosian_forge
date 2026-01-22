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
def _validate_stages(self, stages: Iterable[str]) -> None:
    invalid_stages = [stage for stage in stages if self.invalid_stage_regex.search(stage) is not None]
    if invalid_stages:
        with io.StringIO() as msg:
            msg.write(f'The following stage names are not valid: {invalid_stages[0]}')
            for invalid_stage in invalid_stages[1:]:
                msg.write(f', {invalid_stage}')
            raise ValueError(msg.getvalue())
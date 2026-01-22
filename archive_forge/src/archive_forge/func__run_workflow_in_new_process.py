from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from itertools import chain
from typing import Any
import dill
from qiskit.utils.parallel import parallel_map
from .base_tasks import Task, PassManagerIR
from .exceptions import PassManagerError
from .flow_controllers import FlowControllerLinear
from .compilation_status import PropertySet, WorkflowStatus, PassManagerState
def _run_workflow_in_new_process(program: Any, pass_manager_bin: bytes) -> Any:
    """Run single program optimization in new process.

    Args:
        program: Arbitrary program to optimize.
        pass_manager_bin: Binary of the pass manager with scheduled passes.

    Returns:
          Optimized program.
    """
    return _run_workflow(program=program, pass_manager=dill.loads(pass_manager_bin))
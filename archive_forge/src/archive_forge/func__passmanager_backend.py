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
@abstractmethod
def _passmanager_backend(self, passmanager_ir: PassManagerIR, in_program: Any, **kwargs) -> Any:
    """Convert pass manager IR into output program.

        Args:
            passmanager_ir: Pass manager IR after optimization.
            in_program: The input program, this can be used if you need
                any metadata about the original input for the output.
                It should not be mutated.

        Returns:
            Output program.
        """
    pass
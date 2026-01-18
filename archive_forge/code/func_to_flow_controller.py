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
def to_flow_controller(self) -> FlowControllerLinear:
    """Linearize this manager into a single :class:`.FlowControllerLinear`,
        so that it can be nested inside another pass manager.

        Returns:
            A linearized pass manager.
        """
    flatten_tasks = list(self._flatten_tasks(self._tasks))
    return FlowControllerLinear(flatten_tasks)
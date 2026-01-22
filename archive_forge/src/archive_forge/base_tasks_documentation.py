from __future__ import annotations
import logging
import time
from abc import abstractmethod, ABC
from collections.abc import Iterable, Callable, Generator
from typing import Any
from .compilation_status import RunState, PassManagerState, PropertySet
A custom logic to choose a next task to run.

        Controller subclass can consume the state to build a proper task pipeline.  The updated
        state after a task execution will be fed back in as the "return" value of any ``yield``
        statements.  This indicates the order of task execution is only determined at running time.
        This method is not allowed to mutate the given state object.

        Args:
            state: The state of the passmanager workflow at the beginning of this flow controller's
                execution.

        Receives:
            state: the state of pass manager after the execution of the last task that was yielded.
                The generator does not need to inspect this if it is irrelevant to its logic, nor
                update it.

        Yields:
            Task: Next task to run.
        
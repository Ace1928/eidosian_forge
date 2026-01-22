from __future__ import annotations
import logging
from collections.abc import Callable, Iterable, Generator
from typing import Any
from .base_tasks import BaseController, Task
from .compilation_status import PassManagerState, PropertySet
from .exceptions import PassManagerError
class ConditionalController(BaseController):
    """A flow controller runs the pipeline once if the condition is true, or does nothing if the
    condition is false."""

    def __init__(self, tasks: Task | Iterable[Task]=(), condition: Callable[[PropertySet], bool]=None, *, options: dict[str, Any] | None=None):
        super().__init__(options)
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        self.tasks: tuple[Task] = tuple(tasks)
        self.condition = condition

    @property
    def passes(self) -> list[Task]:
        """Alias of tasks for backward compatibility."""
        return list(self.tasks)

    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        if self.condition(state.property_set):
            for task in self.tasks:
                state = (yield task)
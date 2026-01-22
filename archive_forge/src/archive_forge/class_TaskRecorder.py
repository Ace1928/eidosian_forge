from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
@attrs.define(eq=False, hash=False, slots=False)
class TaskRecorder(_abc.Instrument):
    record: list[tuple[str, Task | None]] = attrs.Factory(list)

    def before_run(self) -> None:
        self.record.append(('before_run', None))

    def task_scheduled(self, task: Task) -> None:
        self.record.append(('schedule', task))

    def before_task_step(self, task: Task) -> None:
        assert task is _core.current_task()
        self.record.append(('before', task))

    def after_task_step(self, task: Task) -> None:
        assert task is _core.current_task()
        self.record.append(('after', task))

    def after_run(self) -> None:
        self.record.append(('after_run', None))

    def filter_tasks(self, tasks: Container[Task]) -> Iterable[tuple[str, Task | None]]:
        for item in self.record:
            if item[0] in ('schedule', 'before', 'after') and item[1] in tasks:
                yield item
            if item[0] in ('before_run', 'after_run'):
                yield item
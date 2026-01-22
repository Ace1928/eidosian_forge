from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
class SpawnExitRecorder(_abc.Instrument):

    def task_spawned(self, task: Task) -> None:
        record.append(('spawned', task))

    def task_exited(self, task: Task) -> None:
        record.append(('exited', task))
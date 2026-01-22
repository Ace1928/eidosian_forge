from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
class BrokenInstrument(_abc.Instrument):

    def task_scheduled(self, task: Task) -> NoReturn:
        record.append('scheduled')
        raise ValueError('oops')

    def close(self) -> None:
        record.append('closed')
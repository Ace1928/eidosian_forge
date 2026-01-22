from __future__ import annotations
from typing import NoReturn, TypeVar
from attrs import define as _define, frozen as _frozen
@staticmethod
def _do_not_subclass() -> NoReturn:
    raise UnsupportedSubclassing()
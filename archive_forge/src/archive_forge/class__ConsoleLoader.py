from __future__ import annotations
import code
import sys
import typing as t
from contextvars import ContextVar
from types import CodeType
from markupsafe import escape
from .repr import debug_repr
from .repr import dump
from .repr import helper
class _ConsoleLoader:

    def __init__(self) -> None:
        self._storage: dict[int, str] = {}

    def register(self, code: CodeType, source: str) -> None:
        self._storage[id(code)] = source
        for var in code.co_consts:
            if isinstance(var, CodeType):
                self._storage[id(var)] = source

    def get_source_by_code(self, code: CodeType) -> str | None:
        try:
            return self._storage[id(code)]
        except KeyError:
            return None
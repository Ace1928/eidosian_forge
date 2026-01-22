from __future__ import annotations
from copy import copy
from typing import Any
from tomlkit.exceptions import ParseError
from tomlkit.exceptions import UnexpectedCharError
from tomlkit.toml_char import TOMLChar
class _StateHandler:
    """
    State preserver for the Parser.
    """

    def __init__(self, source: Source) -> None:
        self._source = source
        self._states = []

    def __call__(self, *args, **kwargs):
        return _State(self._source, *args, **kwargs)

    def __enter__(self) -> _State:
        state = self()
        self._states.append(state)
        return state.__enter__()

    def __exit__(self, exception_type, exception_val, trace):
        state = self._states.pop()
        return state.__exit__(exception_type, exception_val, trace)
from __future__ import annotations
import re
import sys
from typing import (
from trio._util import final
@final
class _ExceptionInfo(Generic[E]):
    """Minimal re-implementation of pytest.ExceptionInfo, only used if pytest is not available. Supports a subset of its features necessary for functionality of :class:`trio.testing.RaisesGroup` and :class:`trio.testing.Matcher`."""
    _excinfo: tuple[type[E], E, types.TracebackType] | None

    def __init__(self, excinfo: tuple[type[E], E, types.TracebackType] | None):
        self._excinfo = excinfo

    def fill_unfilled(self, exc_info: tuple[type[E], E, types.TracebackType]) -> None:
        """Fill an unfilled ExceptionInfo created with ``for_later()``."""
        assert self._excinfo is None, 'ExceptionInfo was already filled'
        self._excinfo = exc_info

    @classmethod
    def for_later(cls) -> _ExceptionInfo[E]:
        """Return an unfilled ExceptionInfo."""
        return cls(None)

    @property
    def type(self) -> type[E]:
        """The exception class."""
        assert self._excinfo is not None, '.type can only be used after the context manager exits'
        return self._excinfo[0]

    @property
    def value(self) -> E:
        """The exception value."""
        assert self._excinfo is not None, '.value can only be used after the context manager exits'
        return self._excinfo[1]

    @property
    def tb(self) -> types.TracebackType:
        """The exception raw traceback."""
        assert self._excinfo is not None, '.tb can only be used after the context manager exits'
        return self._excinfo[2]

    def exconly(self, tryshort: bool=False) -> str:
        raise NotImplementedError('This is a helper method only available if you use RaisesGroup with the pytest package installed')

    def errisinstance(self, exc: builtins.type[BaseException] | tuple[builtins.type[BaseException], ...]) -> bool:
        raise NotImplementedError('This is a helper method only available if you use RaisesGroup with the pytest package installed')

    def getrepr(self, showlocals: bool=False, style: str='long', abspath: bool=False, tbfilter: bool | Callable[[_ExceptionInfo[BaseException]], Traceback]=True, funcargs: bool=False, truncate_locals: bool=True, chain: bool=True) -> ReprExceptionInfo | ExceptionChainRepr:
        raise NotImplementedError('This is a helper method only available if you use RaisesGroup with the pytest package installed')
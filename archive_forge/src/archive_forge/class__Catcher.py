from __future__ import annotations
import inspect
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any
class _Catcher:

    def __init__(self, handler_map: Mapping[tuple[type[BaseException], ...], _Handler]):
        self._handler_map = handler_map

    def __enter__(self) -> None:
        pass

    def __exit__(self, etype: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        if exc is not None:
            unhandled = self.handle_exception(exc)
            if unhandled is exc:
                return False
            elif unhandled is None:
                return True
            else:
                if isinstance(exc, BaseExceptionGroup):
                    try:
                        raise unhandled from exc.__cause__
                    except BaseExceptionGroup:
                        unhandled.__context__ = exc.__cause__
                        raise
                raise unhandled from exc
        return False

    def handle_exception(self, exc: BaseException) -> BaseException | None:
        excgroup: BaseExceptionGroup | None
        if isinstance(exc, BaseExceptionGroup):
            excgroup = exc
        else:
            excgroup = BaseExceptionGroup('', [exc])
        new_exceptions: list[BaseException] = []
        for exc_types, handler in self._handler_map.items():
            matched, excgroup = excgroup.split(exc_types)
            if matched:
                try:
                    try:
                        raise matched
                    except BaseExceptionGroup:
                        result = handler(matched)
                except BaseExceptionGroup as new_exc:
                    if new_exc is matched:
                        new_exceptions.append(new_exc)
                    else:
                        new_exceptions.extend(new_exc.exceptions)
                except BaseException as new_exc:
                    new_exceptions.append(new_exc)
                else:
                    if inspect.iscoroutine(result):
                        raise TypeError(f'Error trying to handle {matched!r} with {handler!r}. Exception handler must be a sync function.') from exc
            if not excgroup:
                break
        if new_exceptions:
            if len(new_exceptions) == 1:
                return new_exceptions[0]
            return BaseExceptionGroup('', new_exceptions)
        elif excgroup and len(excgroup.exceptions) == 1 and (excgroup.exceptions[0] is exc):
            return exc
        else:
            return excgroup
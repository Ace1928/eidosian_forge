from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
class StoppingContext(typing.ContextManager['StoppingContext']):
    """Context manager that calls ``stop`` on a given object on exit.  Used to
    make the ``start`` method on `MainLoop` and `BaseScreen` optionally act as
    context managers.
    """
    __slots__ = ('_wrapped',)

    def __init__(self, wrapped: CanBeStopped) -> None:
        self._wrapped = wrapped

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        self._wrapped.stop()
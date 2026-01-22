from __future__ import annotations
import os
import pathlib
import sys
from functools import partial, update_wrapper
from inspect import cleandoc
from typing import IO, TYPE_CHECKING, Any, BinaryIO, ClassVar, TypeVar, overload
from trio._file_io import AsyncIOWrapper, wrap_file
from trio._util import final
from trio.to_thread import run_sync
def _wrap_method_path_iterable(fn: Callable[Concatenate[pathlib.Path, P], Iterable[pathlib.Path]]) -> Callable[Concatenate[PathT, P], Awaitable[Iterable[PathT]]]:

    @_wraps_async(fn)
    def wrapper(self: PathT, /, *args: P.args, **kwargs: P.kwargs) -> Iterable[PathT]:
        return map(self.__class__, [*fn(self._wrapped_cls(self), *args, **kwargs)])
    assert wrapper.__doc__ is not None
    wrapper.__doc__ += f'\nThis is an async method that returns a synchronous iterator, so you\nuse it like:\n\n.. code:: python\n\n    for subpath in await mypath.{fn.__name__}():\n        ...\n\n.. note::\n\n    The iterator is loaded into memory immediately during the initial\n    call (see `issue #501\n    <https://github.com/python-trio/trio/issues/501>`__ for discussion).\n'
    return wrapper
from __future__ import annotations
import typing as t
from datetime import datetime
from markupsafe import escape
from markupsafe import Markup
from ._internal import _get_environ
class Aborter:
    """When passed a dict of code -> exception items it can be used as
    callable that raises exceptions.  If the first argument to the
    callable is an integer it will be looked up in the mapping, if it's
    a WSGI application it will be raised in a proxy exception.

    The rest of the arguments are forwarded to the exception constructor.
    """

    def __init__(self, mapping: dict[int, type[HTTPException]] | None=None, extra: dict[int, type[HTTPException]] | None=None) -> None:
        if mapping is None:
            mapping = default_exceptions
        self.mapping = dict(mapping)
        if extra is not None:
            self.mapping.update(extra)

    def __call__(self, code: int | Response, *args: t.Any, **kwargs: t.Any) -> t.NoReturn:
        from .sansio.response import Response
        if isinstance(code, Response):
            raise HTTPException(response=code)
        if code not in self.mapping:
            raise LookupError(f'no exception for {code!r}')
        raise self.mapping[code](*args, **kwargs)
from __future__ import annotations
import typing as t
from datetime import datetime
from markupsafe import escape
from markupsafe import Markup
from ._internal import _get_environ
class BadRequestKeyError(BadRequest, KeyError):
    """An exception that is used to signal both a :exc:`KeyError` and a
    :exc:`BadRequest`. Used by many of the datastructures.
    """
    _description = BadRequest.description
    show_exception = False

    def __init__(self, arg: str | None=None, *args: t.Any, **kwargs: t.Any):
        super().__init__(*args, **kwargs)
        if arg is None:
            KeyError.__init__(self)
        else:
            KeyError.__init__(self, arg)

    @property
    def description(self) -> str:
        if self.show_exception:
            return f'{self._description}\n{KeyError.__name__}: {KeyError.__str__(self)}'
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        self._description = value
from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
class Unicode(TraitType[G, S]):
    """A trait for unicode strings."""
    default_value = ''
    info_text = 'a unicode string'
    if t.TYPE_CHECKING:

        @t.overload
        def __init__(self: Unicode[str, str | bytes], default_value: str | Sentinel=..., allow_none: Literal[False]=..., read_only: bool | None=..., help: str | None=..., config: t.Any=..., **kwargs: t.Any) -> None:
            ...

        @t.overload
        def __init__(self: Unicode[str | None, str | bytes | None], default_value: str | Sentinel | None=..., allow_none: Literal[True]=..., read_only: bool | None=..., help: str | None=..., config: t.Any=..., **kwargs: t.Any) -> None:
            ...

        def __init__(self: Unicode[str | None, str | bytes | None], default_value: str | Sentinel | None=..., allow_none: bool=..., read_only: bool | None=..., help: str | None=..., config: t.Any=..., **kwargs: t.Any) -> None:
            ...

    def validate(self, obj: t.Any, value: t.Any) -> G:
        if isinstance(value, str):
            return t.cast(G, value)
        if isinstance(value, bytes):
            try:
                return t.cast(G, value.decode('ascii', 'strict'))
            except UnicodeDecodeError as e:
                msg = "Could not decode {!r} for unicode trait '{}' of {} instance."
                raise TraitError(msg.format(value, self.name, class_of(obj))) from e
        self.error(obj, value)

    def from_string(self, s: str) -> G:
        if self.allow_none and s == 'None':
            return t.cast(G, None)
        s = os.path.expanduser(s)
        if len(s) >= 2:
            for c in ('"', "'"):
                if s[0] == s[-1] == c:
                    old_s = s
                    s = s[1:-1]
                    warn(f'Supporting extra quotes around strings is deprecated in traitlets 5.0. You can use {s!r} instead of {old_s!r} if you require traitlets >=5.', DeprecationWarning, stacklevel=2)
        return t.cast(G, s)

    def subclass_init(self, cls: type[t.Any]) -> None:
        pass
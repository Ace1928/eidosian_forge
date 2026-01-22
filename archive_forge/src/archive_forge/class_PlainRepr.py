from __future__ import annotations as _annotations
import types
import typing
from typing import Any
import typing_extensions
from . import _typing_extra
class PlainRepr(str):
    """String class where repr doesn't include quotes. Useful with Representation when you want to return a string
    representation of something that is valid (or pseudo-valid) python.
    """

    def __repr__(self) -> str:
        return str(self)
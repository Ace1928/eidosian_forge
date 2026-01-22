from __future__ import annotations
from typing import TYPE_CHECKING, Any
import attrs
from referencing._attrs import frozen
@frozen
class PointerToNowhere(Unresolvable):
    """
    A JSON Pointer leads to a part of a document that does not exist.
    """
    resource: Resource[Any]

    def __str__(self) -> str:
        msg = f'{self.ref!r} does not exist within {self.resource.contents!r}'
        if self.ref == '/':
            msg += ". The pointer '/' is a valid JSON Pointer but it points to an empty string property ''. If you intended to point to the entire resource, you should use '#'."
        return msg
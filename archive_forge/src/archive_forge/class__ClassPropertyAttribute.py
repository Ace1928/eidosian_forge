from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar, cast, overload
class _ClassPropertyAttribute(Protocol[_T]):

    def __get__(self, obj: object, objtype: type[Any] | None=None) -> _T:
        ...

    def __set__(self, obj: object, value: _T) -> None:
        ...
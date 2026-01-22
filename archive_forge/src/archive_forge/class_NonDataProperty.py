from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar, cast, overload
class NonDataProperty:
    """Much like the property builtin, but only implements __get__,
    making it a non-data property, and can be subsequently reset.

    See http://users.rcn.com/python/download/Descriptor.htm for more
    information.

    >>> class X(object):
    ...   @NonDataProperty
    ...   def foo(self):
    ...     return 3
    >>> x = X()
    >>> x.foo
    3
    >>> x.foo = 4
    >>> x.foo
    4

    '...' below should be 'jaraco.classes' but for pytest-dev/pytest#3396
    >>> X.foo
    <....properties.NonDataProperty object at ...>
    """

    def __init__(self, fget: Callable[[object], object]) -> None:
        assert fget is not None, 'fget cannot be none'
        assert callable(fget), 'fget must be callable'
        self.fget = fget

    @overload
    def __get__(self, obj: None, objtype: type[object] | None=None) -> Self:
        ...

    @overload
    def __get__(self, obj: object, objtype: type[object] | None=None) -> object:
        ...

    def __get__(self, obj: object | None, objtype: type[object] | None=None) -> Self | object:
        if obj is None:
            return self
        return self.fget(obj)
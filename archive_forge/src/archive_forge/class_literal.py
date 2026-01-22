from __future__ import annotations
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Literal, TypeVar, cast, overload
from dask.typing import Graph, Key, NoDefault, no_default
class literal:
    """A small serializable object to wrap literal values without copying"""
    __slots__ = ('data',)

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return 'literal<type=%s>' % type(self.data).__name__

    def __reduce__(self):
        return (literal, (self.data,))

    def __call__(self):
        return self.data
from __future__ import annotations
from itertools import repeat
from .._internal import _missing
class ImmutableHeadersMixin:
    """Makes a :class:`Headers` immutable.  We do not mark them as
    hashable though since the only usecase for this datastructure
    in Werkzeug is a view on a mutable structure.

    .. versionadded:: 0.5

    :private:
    """

    def __delitem__(self, key, **kwargs):
        is_immutable(self)

    def __setitem__(self, key, value):
        is_immutable(self)

    def set(self, _key, _value, **kwargs):
        is_immutable(self)

    def setlist(self, key, values):
        is_immutable(self)

    def add(self, _key, _value, **kwargs):
        is_immutable(self)

    def add_header(self, _key, _value, **_kwargs):
        is_immutable(self)

    def remove(self, key):
        is_immutable(self)

    def extend(self, *args, **kwargs):
        is_immutable(self)

    def update(self, *args, **kwargs):
        is_immutable(self)

    def insert(self, pos, value):
        is_immutable(self)

    def pop(self, key=None, default=_missing):
        is_immutable(self)

    def popitem(self):
        is_immutable(self)

    def setdefault(self, key, default):
        is_immutable(self)

    def setlistdefault(self, key, default):
        is_immutable(self)
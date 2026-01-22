from __future__ import annotations
from collections.abc import MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
@dataclass
class StackedLookup(MutableMapping):
    """
    Iterative lookup in a stack of dicts

    Assignments go into an internal dict that is also the first place
    where a lookup is done.
    """
    stack: list[SupportsGetItem]

    def __post_init__(self):
        self._dict = {}
        self.stack = [self._dict] + list(self.stack)

    def __getitem__(self, key):
        for d in self.stack:
            with suppress(KeyError):
                return d[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __contains__(self, key):
        with suppress(KeyError):
            self[key]
            return True
        return False

    def __iter__(self):
        """
        Unique keys in stacking order
        """
        return iter({key: None for d in self.stack for key in d})

    def __len__(self):
        """
        Number of unique keys
        """
        return len({key for d in self.stack for key in d})

    def get(self, key: str, default: Any=None) -> Any:
        with suppress(KeyError):
            return self[key]
        return default

    def __repr__(self):
        return f'{self.__class__.__name__}({self.stack})'

    def __getstate__(*args, **kwargs):
        """
        Return state with no namespace
        """
        d = {}
        return {'stack': [d], '_dict': d}

    def copy(self):
        return self

    def __deepcopy__(self, memo: dict[Any, Any]) -> StackedLookup:
        """
        Shallow copy
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        old = self.__dict__
        new = result.__dict__
        for key, item in old.items():
            new[key] = item
            memo[id(new[key])] = new[key]
        return result
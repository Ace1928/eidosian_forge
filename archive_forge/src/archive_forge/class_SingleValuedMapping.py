from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
class SingleValuedMapping(t.Mapping[K, V]):
    """
    Mapping where all keys return the same value.

    This rigamarole is meant to avoid copying keys, which was originally intended
    as an optimization while qualifying columns for tables with lots of columns.
    """

    def __init__(self, keys: t.Collection[K], value: V):
        self._keys = keys if isinstance(keys, Set) else set(keys)
        self._value = value

    def __getitem__(self, key: K) -> V:
        if key in self._keys:
            return self._value
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> t.Iterator[K]:
        return iter(self._keys)
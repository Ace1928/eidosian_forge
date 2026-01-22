import abc
import asyncio
import datetime
import functools
import importlib
import json
import logging
import os
import pkgutil
from abc import ABCMeta, abstractmethod
from base64 import b64decode
from collections import namedtuple
from collections.abc import MutableMapping, Mapping, Sequence
from typing import Optional
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._raylet import GcsClient
from ray._private.utils import split_address
import aiosignal  # noqa: F401
import ray._private.protobuf_compat
from frozenlist import FrozenList  # noqa: F401
from ray._private.utils import binary_to_hex, check_dashboard_dependencies_installed
class ImmutableList(Immutable, Sequence):
    """Makes a :class:`list` immutable."""
    __slots__ = ('_list', '_proxy')

    def __init__(self, list_value):
        if type(list_value) not in (list, ImmutableList):
            raise TypeError(f'{type(list_value)} object is not a list.')
        if isinstance(list_value, ImmutableList):
            list_value = list_value.mutable()
        self._list = list_value
        self._proxy = [None] * len(list_value)

    def __reduce_ex__(self, protocol):
        return (type(self), (self._list,))

    def mutable(self):
        return self._list

    def __eq__(self, other):
        if isinstance(other, ImmutableList):
            other = other.mutable()
        return list.__eq__(self._list, other)

    def __ne__(self, other):
        if isinstance(other, ImmutableList):
            other = other.mutable()
        return list.__ne__(self._list, other)

    def __contains__(self, item):
        if isinstance(item, Immutable):
            item = item.mutable()
        return list.__contains__(self._list, item)

    def __getitem__(self, item):
        proxy = self._proxy[item]
        if proxy is None:
            proxy = self._proxy[item] = make_immutable(self._list[item])
        return proxy

    def __len__(self):
        return len(self._list)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, list.__repr__(self._list))
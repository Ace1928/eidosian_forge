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
class MutableNotificationDict(dict, MutableMapping):
    """A simple descriptor for dict type to notify data changes.
    :note: Only the first level data report change.
    """
    ChangeItem = namedtuple('DictChangeItem', ['key', 'value'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signal = Signal(self)

    def mutable(self):
        return self

    @property
    def signal(self):
        return self._signal

    def __setitem__(self, key, value):
        old = self.pop(key, None)
        super().__setitem__(key, value)
        if len(self._signal) and old != value:
            if old is None:
                co = self._signal.send(Change(owner=self, new=Dict.ChangeItem(key, value)))
            else:
                co = self._signal.send(Change(owner=self, old=Dict.ChangeItem(key, old), new=Dict.ChangeItem(key, value)))
            NotifyQueue.put(co)

    def __delitem__(self, key):
        old = self.pop(key, None)
        if len(self._signal) and old is not None:
            co = self._signal.send(Change(owner=self, old=Dict.ChangeItem(key, old)))
            NotifyQueue.put(co)

    def reset(self, d):
        assert isinstance(d, Mapping)
        for key in self.keys() - d.keys():
            del self[key]
        for key, value in d.items():
            self[key] = value
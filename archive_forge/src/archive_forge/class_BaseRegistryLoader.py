from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
class BaseRegistryLoader(object):
    """
    An incremental loader for a registry.  Each new call to
    new_registrations() will iterate over the not yet seen registrations.

    The reason for this object is multiple:
    - there can be several contexts
    - each context wants to install all registrations
    - registrations can be added after the first installation, so contexts
      must be able to get the "new" installations

    Therefore each context maintains its own loaders for each existing
    registry, without duplicating the registries themselves.
    """

    def __init__(self, registry):
        self._registrations = dict(((name, utils.stream_list(getattr(registry, name))) for name in self.registry_items))

    def new_registrations(self, name):
        for item in next(self._registrations[name]):
            yield item
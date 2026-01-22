import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def _get_instance_from_parent(self, key: Type[T], provider: Provider[T], parent: 'Injector') -> T:
    singleton_scope_binding, _ = parent.binder.get_binding(type(self))
    singleton_scope = singleton_scope_binding.provider.get(parent)
    provider = singleton_scope.get(key, provider)
    return provider.get(parent)
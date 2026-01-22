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
def _get_binding(self, key: type, *, only_this_binder: bool=False) -> Tuple[Binding, 'Binder']:
    binding = self._bindings.get(key)
    if binding:
        return (binding, self)
    if self.parent and (not only_this_binder):
        return self.parent._get_binding(key)
    raise KeyError
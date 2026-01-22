import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional
from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
class MutationAwareDict(py_collections.OrderedDict):
    """A dict with a mutation flag."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mutated = True

    def pop(self, key, default=None):
        self._mutated = True
        return super().pop(key, default)

    def __setitem__(self, key, value):
        self._mutated = True
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        self._mutated = True
        return super().__delitem__(key)

    def clear(self):
        self._mutated = True
        return super().clear()

    @property
    def mutated(self):
        return self._mutated

    @mutated.setter
    def mutated(self, value):
        self._mutated = value
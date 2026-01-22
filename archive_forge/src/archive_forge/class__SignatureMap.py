from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import attributes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.trackable import base
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class _SignatureMap(collections_abc.Mapping, base.Trackable):
    """A collection of SavedModel signatures."""

    def __init__(self):
        self._signatures = {}

    def _add_signature(self, name, concrete_function):
        """Adds a signature to the _SignatureMap."""
        self._signatures[name] = concrete_function

    def __getitem__(self, key):
        return self._signatures[key]

    def __iter__(self):
        return iter(self._signatures)

    def __len__(self):
        return len(self._signatures)

    def __repr__(self):
        return '_SignatureMap({})'.format(self._signatures)

    def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
        if save_type != base.SaveType.SAVEDMODEL:
            return {}
        return {key: value for key, value in self.items() if isinstance(value, (def_function.Function, defun.ConcreteFunction))}
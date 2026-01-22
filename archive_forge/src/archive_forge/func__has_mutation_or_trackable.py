import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _has_mutation_or_trackable(self):
    """Short-circuits a check for trackables if there's already a mutation."""
    if self._non_append_mutation:
        return True
    return any((isinstance(element, base.Trackable) for element in self._storage))
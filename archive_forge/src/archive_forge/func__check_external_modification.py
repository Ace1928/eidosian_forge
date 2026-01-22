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
def _check_external_modification(self):
    """Checks for any changes to the wrapped list not through the wrapper."""
    if self._external_modification or self._non_append_mutation:
        return
    if self._storage != self._last_wrapped_list_snapshot:
        self._external_modification = True
        self._last_wrapped_list_snapshot = None
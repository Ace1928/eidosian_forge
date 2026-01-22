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
@property
def _layers(self):
    """All Layers and Layer containers, including empty containers."""
    collected = []
    for obj in self._values:
        if isinstance(obj, TrackableDataStructure) or layer_utils.is_layer(obj) or layer_utils.has_weights(obj):
            collected.append(obj)
    return collected
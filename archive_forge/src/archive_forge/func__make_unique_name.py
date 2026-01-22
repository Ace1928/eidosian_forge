import copy
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.legacy_tf_layers import variable_scope_shim
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def _make_unique_name(self, name_uid_map=None, avoid_names=None, namespace='', zero_based=False):
    base_name = base_layer.to_snake_case(self.__class__.__name__)
    name = backend.unique_object_name(base_name, name_uid_map=name_uid_map, avoid_names=avoid_names, namespace=namespace, zero_based=zero_based)
    return (name, base_name)
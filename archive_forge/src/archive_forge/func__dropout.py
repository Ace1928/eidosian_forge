import hashlib
import numbers
import sys
import types as python_types
import warnings
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
def _dropout(self, values, salt_prefix, recurrent_noise, keep_prob, shallow_filtered_substructure=None):
    """Decides whether to perform standard dropout or recurrent dropout."""
    if shallow_filtered_substructure is None:
        shallow_filtered_substructure = values
    if not self._variational_recurrent:

        def dropout(i, do_dropout, v):
            if not isinstance(do_dropout, bool) or do_dropout:
                return nn_ops.dropout_v2(v, rate=1.0 - keep_prob, seed=self._gen_seed(salt_prefix, i))
            else:
                return v
        return _enumerated_map_structure_up_to(shallow_filtered_substructure, dropout, *[shallow_filtered_substructure, values])
    else:

        def dropout(i, do_dropout, v, n):
            if not isinstance(do_dropout, bool) or do_dropout:
                return self._variational_recurrent_dropout_value(i, v, n, keep_prob)
            else:
                return v
        return _enumerated_map_structure_up_to(shallow_filtered_substructure, dropout, *[shallow_filtered_substructure, values, recurrent_noise])
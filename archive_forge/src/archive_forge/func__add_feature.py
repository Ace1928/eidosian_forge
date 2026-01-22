import collections
import re
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import tf_export
def _add_feature(self, key, feature):
    """Adds the specified feature to this ParseOpParams."""
    if isinstance(feature, VarLenFeature):
        self._add_varlen_feature(key, feature)
    elif isinstance(feature, SparseFeature):
        self._add_sparse_feature(key, feature)
    elif isinstance(feature, FixedLenFeature):
        self._add_fixed_len_feature(key, feature)
    elif isinstance(feature, FixedLenSequenceFeature):
        self._add_fixed_len_sequence_feature(key, feature)
    elif isinstance(feature, RaggedFeature):
        self._add_ragged_feature(key, feature)
    else:
        raise ValueError(f'Invalid feature {key}:{feature}.')
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
@tf_export('io.VarLenFeature', v1=['VarLenFeature', 'io.VarLenFeature'])
class VarLenFeature(collections.namedtuple('VarLenFeature', ['dtype'])):
    """Configuration for parsing a variable-length input feature.

  Fields:
    dtype: Data type of input.
  """
    pass
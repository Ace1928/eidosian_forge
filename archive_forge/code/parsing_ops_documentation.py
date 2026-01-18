from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
Function from `Dataset` to `Dataset` that applies the transformation.
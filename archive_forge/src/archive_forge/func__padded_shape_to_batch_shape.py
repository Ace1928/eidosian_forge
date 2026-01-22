import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
def _padded_shape_to_batch_shape(s):
    return tensor_shape.TensorShape([tensor_util.constant_value(self._batch_size) if smart_cond.smart_constant_value(self._drop_remainder) else None]).concatenate(tensor_util.constant_value_as_shape(s))
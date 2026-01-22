import os
import numpy as np
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import nest
def _get_iterator_ops_from_collection(self, ds_fn, sparse_tensors=False):
    all_ops = ops.get_collection('iterator_ops')
    if sparse_tensors:
        init_op, indices, values, dense_shape = all_ops
        return (init_op, sparse_tensor.SparseTensor(indices, values, dense_shape))
    get_next_list = []
    i = 1
    for output_class in nest.flatten(self._get_output_classes(ds_fn)):
        if output_class is sparse_tensor.SparseTensor:
            indices, values, dense_shape = all_ops[i:i + 3]
            i += 3
            get_next_list.append(sparse_tensor.SparseTensor(indices, values, dense_shape))
        else:
            get_next_list.append(all_ops[i])
            i += 1
    return (all_ops[0], nest.pack_sequence_as(self._get_output_types(ds_fn), get_next_list))
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
def _add_iterator_ops_to_collection(self, init_op, get_next, ds_fn, sparse_tensors=False):
    ops.add_to_collection('iterator_ops', init_op)
    if sparse_tensors:
        ops.add_to_collection('iterator_ops', get_next.indices)
        ops.add_to_collection('iterator_ops', get_next.values)
        ops.add_to_collection('iterator_ops', get_next.dense_shape)
        return
    get_next_list = nest.flatten(get_next)
    for i, output_class in enumerate(nest.flatten(self._get_output_classes(ds_fn))):
        if output_class is sparse_tensor.SparseTensor:
            ops.add_to_collection('iterator_ops', get_next_list[i].indices)
            ops.add_to_collection('iterator_ops', get_next_list[i].values)
            ops.add_to_collection('iterator_ops', get_next_list[i].dense_shape)
        else:
            ops.add_to_collection('iterator_ops', get_next_list[i])
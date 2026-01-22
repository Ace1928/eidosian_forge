from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _store_sparse_tensors_join(tensor_list_list, enqueue_many, keep_input):
    """Store SparseTensors for feeding into batch_join, etc."""
    s0, sparse_info_list = _store_sparse_tensors(tensor_list_list[0], enqueue_many, keep_input)
    stored_list_list = [s0]
    for tensor_list in tensor_list_list[1:]:
        s, sparse_info_candidate = _store_sparse_tensors(tensor_list, enqueue_many, keep_input, [st.map_op for st in sparse_info_list])
        if sparse_info_list != sparse_info_candidate:
            raise ValueError('Inconsistent SparseTensors list: %s vs. %s' % (tensor_list_list[0], tensor_list))
        sparse_info_list = [info.merge_with(candidate) for info, candidate in zip(sparse_info_list, sparse_info_candidate)]
        stored_list_list.append(s)
    return (stored_list_list, sparse_info_list)
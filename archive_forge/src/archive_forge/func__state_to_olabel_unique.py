import uuid
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _state_to_olabel_unique(labels, num_labels, states, unique):
    """Sum state log probs to ilabel log probs using unique label indices."""
    num_label_states = _get_dim(labels, 1) + 1
    label_states = states[:, :, 1:num_label_states]
    blank_states = states[:, :, num_label_states:]
    unique_y, unique_idx = unique
    mul_reduce = _sum_states(unique_idx, label_states)
    num_frames = _get_dim(states, 0)
    batch_size = _get_dim(states, 1)
    num_states = num_label_states - 1
    batch_state_major = array_ops.transpose(mul_reduce, perm=[1, 2, 0])
    batch_state_major = array_ops.reshape(batch_state_major, [batch_size * num_states, num_frames])
    batch_offset = math_ops.range(batch_size, dtype=unique_y.dtype) * num_labels
    indices = unique_y + array_ops.expand_dims(batch_offset, axis=-1)
    indices = array_ops.reshape(indices, [-1, 1])
    scatter = array_ops.scatter_nd(indices=indices, updates=batch_state_major, shape=[batch_size * num_labels, num_frames])
    scatter = array_ops.reshape(scatter, [batch_size, num_labels, num_frames])
    mask = array_ops.ones_like(batch_state_major, dtype=dtypes.bool)
    mask = array_ops.scatter_nd(indices=indices, updates=mask, shape=[batch_size * num_labels, num_frames])
    mask = array_ops.reshape(mask, [batch_size, num_labels, num_frames])
    scatter = array_ops.where(mask, scatter, array_ops.fill(array_ops.shape(scatter), math_ops.log(0.0)))
    label_olabels = array_ops.transpose(scatter, [2, 0, 1])
    label_olabels = label_olabels[:, :, 1:]
    blank_olabels = math_ops.reduce_logsumexp(blank_states, axis=2, keepdims=True)
    return array_ops.concat([blank_olabels, label_olabels], axis=-1)
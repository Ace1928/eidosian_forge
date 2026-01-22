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
def _ctc_loss_impl(labels, inputs=None, sequence_length=None, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=True, logits=None, use_cudnn=False):
    if not isinstance(labels, sparse_tensor.SparseTensor):
        raise TypeError(f'Expected argument `labels` to be a SparseTensor. Received labels={labels} of type: {type(labels).__name__}')
    inputs = deprecation.deprecated_argument_lookup('logits', logits, 'inputs', inputs)
    inputs = ops.convert_to_tensor(inputs, name='logits')
    if not time_major:
        inputs = array_ops.transpose(inputs, [1, 0, 2])
    orig_dtype = inputs.dtype
    if orig_dtype in (dtypes.float16, dtypes.bfloat16):
        inputs = math_ops.cast(inputs, dtypes.float32)
    if use_cudnn:
        ctc_loss_func = gen_ctc_ops.ctc_loss_v2
    else:
        ctc_loss_func = gen_ctc_ops.ctc_loss
    loss, _ = ctc_loss_func(inputs, labels.indices, labels.values, sequence_length, preprocess_collapse_repeated=preprocess_collapse_repeated, ctc_merge_repeated=ctc_merge_repeated, ignore_longer_outputs_than_inputs=ignore_longer_outputs_than_inputs)
    if orig_dtype in (dtypes.float16, dtypes.bfloat16):
        loss = math_ops.cast(loss, orig_dtype)
    return loss
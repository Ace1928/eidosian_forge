from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('ExtractImagePatches')
def _ExtractImagePatchesGrad(op, grad):
    input_bhwc = array_ops.shape(op.inputs[0], out_type=dtypes.int64)
    batch_size, rows_in, cols_in, channels = array_ops_stack.unstack(input_bhwc)
    output_bhwc = array_ops.shape(op.outputs[0], out_type=dtypes.int64)
    rows_out, cols_out = array_ops_stack.unstack(output_bhwc[1:3])
    _, ksize_r, ksize_c, _ = op.get_attr('ksizes')
    input_indices_num = rows_in * cols_in
    input_idx = array_ops.reshape(math_ops.range(1, input_indices_num + 1, dtype=ops.dtypes.float32), (1, rows_in, cols_in, 1))
    input_idx_patched = gen_array_ops.extract_image_patches(input_idx, op.get_attr('ksizes'), op.get_attr('strides'), op.get_attr('rates'), op.get_attr('padding'))
    input_idx_patched = math_ops.cast(input_idx_patched, dtypes.int64)
    grad_expanded = array_ops.transpose(array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), (batch_size, rows_out, cols_out, ksize_r, ksize_c, channels)), (1, 2, 3, 4, 0, 5))
    grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))
    segment_ids = array_ops.reshape(input_idx_patched, [-1]) - 1
    grad_out = math_ops.unsorted_segment_sum(grad_flat, segment_ids, num_segments=input_indices_num)
    grad_out = array_ops.reshape(grad_out, (rows_in, cols_in, batch_size, channels))
    grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))
    return [grad_out]
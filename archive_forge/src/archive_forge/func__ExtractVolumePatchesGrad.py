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
@ops.RegisterGradient('ExtractVolumePatches')
def _ExtractVolumePatchesGrad(op, grad):
    batch_size, planes_in, rows_in, cols_in, channels = [dim.value for dim in op.inputs[0].shape.dims]
    input_bphwc = array_ops.shape(op.inputs[0])
    batch_size = input_bphwc[0]
    channels = input_bphwc[4]
    input_indices_num = 1 + planes_in * rows_in * cols_in
    input_idx = array_ops.reshape(math_ops.range(1, input_indices_num, dtype=ops.dtypes.int64), (1, planes_in, rows_in, cols_in, 1))
    input_idx_patched = gen_array_ops.extract_volume_patches(input_idx, op.get_attr('ksizes'), op.get_attr('strides'), op.get_attr('padding'))
    _, planes_out, rows_out, cols_out, _ = [dim.value for dim in op.outputs[0].shape.dims]
    _, ksize_p, ksize_r, ksize_c, _ = op.get_attr('ksizes')
    prc_indices_num = planes_out * rows_out * cols_out
    output_indices_num = prc_indices_num * ksize_p * ksize_r * ksize_c
    output_idx = array_ops.reshape(math_ops.range(output_indices_num, dtype=ops.dtypes.int64), (1, planes_out, rows_out, cols_out, ksize_p * ksize_r * ksize_c))
    idx_matrix = array_ops.concat([array_ops.expand_dims(input_idx_patched, axis=-1), array_ops.expand_dims(output_idx, axis=-1)], axis=-1)
    idx_map = array_ops.reshape(idx_matrix, (-1, 2))
    sp_shape = (input_indices_num, output_indices_num)
    sp_mat_full = sparse_tensor.SparseTensor(idx_map, array_ops.ones([output_indices_num], dtype=grad.dtype), sp_shape)
    sp_mat = sparse_ops.sparse_slice(sp_mat_full, (1, 0), (input_indices_num - 1, output_indices_num))
    grad_expanded = array_ops.transpose(array_ops.reshape(_IndexedSlicesToTensorNoWarning(grad), (batch_size, planes_out, rows_out, cols_out, ksize_p, ksize_r, ksize_c, channels)), (1, 2, 3, 4, 5, 6, 0, 7))
    grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))
    jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)
    grad_out = array_ops.reshape(jac, (planes_in, rows_in, cols_in, batch_size, channels))
    grad_out = array_ops.transpose(grad_out, (3, 0, 1, 2, 4))
    return [grad_out]
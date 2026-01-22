import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
def _col2im_shape_check(X, output_shape, kernel_shape, dilations, pads, strides):
    n_input_plane = X.shape[0]
    kernel_size = np.prod(kernel_shape)
    if n_input_plane % kernel_size != 0:
        raise ValueError(f"Expected size of input's dimension 1 to be divisible by the product of kernel_size={kernel_size}, but got input.size(1)={n_input_plane} and kernel_shape={kernel_shape}, X.shape={X.shape}, output_shape={output_shape}.")
    input_length = X.shape[1]
    n_dims = len(output_shape)
    n_blocks = []
    for i in range(n_dims):
        n_block = (output_shape[i] + pads[i, :].sum() - dilations[i] * (kernel_shape[i] - 1) - 1) // strides[i] + 1
        n_blocks.append(n_block)
    block_size = np.prod(n_blocks)
    if input_length != block_size:
        raise ValueError(f"Given n_input_plane={n_input_plane}, X.shape={X.shape}, output_shape={output_shape}, kernel_shape={kernel_shape}, dilations={dilations}, pads={pads}, strides={strides}, expected size of input's dimension 2 to match the calculated number of sliding blocks {n_blocks} = {block_size}, but got input.size(2)={input_length}.")
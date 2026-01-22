import itertools
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_indices import _get_index, _get_indices
class CommonPool(OpRun):

    def _run(self, pooling_type, count_include_pad, x, auto_pad=None, ceil_mode=None, dilations=None, kernel_shape=None, pads=None, storage_order=None, strides=None):
        if pooling_type == 'MAX' and dilations is None:
            dilations = [1 for s in kernel_shape]
        if pads is None:
            pads = [0 for s in kernel_shape] * 2
        if strides is None or len(strides) == 0:
            strides = [1] * (len(x.shape) - 2)
        kernel_shape = list(kernel_shape)
        auto_pad = 'VALID' if auto_pad == 'NOTSET' else auto_pad
        if pads is None or len(pads) == 0:
            pad_shape = [0] * (len(x.shape) - 2)
            x_shape = x.shape[2:]
            padded = x
        elif len(pads) == 4:
            pad_top, pad_bottom, pad_left, pad_right = pads
            pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
            x_shape = np.array(x.shape[2:]) + np.array(pad_shape)
            const = np.nan if count_include_pad == 0 else 0
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=const)
        else:
            pad_shape = pads
            x_shape = x.shape[2:]
            padded = x
        if auto_pad in ('SAME_LOWER', 'SAME_UPPER'):
            const = np.nan if count_include_pad == 0 else 0
            out_shape = _get_output_shape(auto_pad, x_shape, kernel_shape, strides, pad_shape, ceil_mode)
            pad_shape = _get_pad_shape(auto_pad, x_shape, kernel_shape, strides, out_shape)
            if auto_pad == 'SAME_LOWER':
                pad_bottom = pad_shape[0] // 2
                pad_top = pad_shape[0] - pad_bottom
                pad_right = pad_shape[1] // 2
                pad_left = pad_shape[1] - pad_right
            else:
                pad_top = pad_shape[0] // 2
                pad_bottom = pad_shape[0] - pad_top
                pad_left = pad_shape[1] // 2
                pad_right = pad_shape[1] - pad_left
            padded = np.pad(padded, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=const)
        else:
            out_shape = _get_output_shape(auto_pad, x_shape, kernel_shape, strides, pad_shape, ceil_mode)
        n_dims = len(pads) // 2
        new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
        res = _pool(padded, x.shape, kernel_shape, strides, out_shape, pad_shape, pooling_type, count_include_pad=count_include_pad, ceil_mode=ceil_mode, indices=len(self.output) > 1, pads=new_pads)
        if isinstance(res, tuple):
            return res
        return (res,)
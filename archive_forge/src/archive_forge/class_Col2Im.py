import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
class Col2Im(OpRun):

    def _run(self, data, image_shape, block_shape, dilations=None, pads=None, strides=None):
        if dilations is None:
            dilations = [1 for s in image_shape]
        if pads is None:
            pads = [0 for s in image_shape] * 2
        if strides is None:
            strides = [1 for s in image_shape]
        bl = np.prod(block_shape)
        C = data.shape[1] // bl
        data = data.reshape(data.shape[:1] + (C,) + (bl,) + data.shape[2:])
        ks = tuple(block_shape)
        res = None
        for n in range(data.shape[0]):
            for c in range(data.shape[1]):
                out = col2im_naive_implementation(data[n, c, ...], image_shape, ks, dilations, pads, strides)
                if res is None:
                    new_shape = data.shape[:2] + out.shape
                    res = np.empty(new_shape, dtype=data.dtype)
                res[n, c, ...] = out
        return (res,)
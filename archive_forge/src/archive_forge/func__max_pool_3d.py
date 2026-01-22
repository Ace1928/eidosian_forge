import numpy as np
from onnx.reference.ops._op_common_pool import CommonPool
def _max_pool_3d(self, x, auto_pad, ceil_mode, dilations, kernel_shape, new_pads, storage_order, strides, output_spatial_shape):
    global_pooling = False
    y_dims = x.shape[:2] + tuple(output_spatial_shape)
    y = np.zeros(y_dims, dtype=x.dtype)
    indices = np.full(y_dims, dtype=np.int64, fill_value=-1)
    x_dims = x.shape
    channels = x_dims[1]
    height = x_dims[2]
    width = x_dims[3] if len(kernel_shape) > 1 else 1
    depth = x_dims[4] if len(kernel_shape) > 2 else 1
    pooled_height = y_dims[2]
    pooled_width = y_dims[3] if len(kernel_shape) > 1 else 1
    pooled_depth = y_dims[4] if len(kernel_shape) > 2 else 1
    total_channels = x_dims[0] * channels
    stride_h = 1 if global_pooling else strides[0]
    stride_w = 1 if global_pooling else strides[1]
    stride_d = 1 if global_pooling else strides[2]
    x_step = height * width * depth
    y_step = pooled_height * pooled_width * pooled_depth
    dilation_h = dilations[0]
    dilation_w = dilations[1]
    dilation_d = dilations[2]
    X_data = x.ravel()
    Y_data = y.ravel()
    I_data = indices.ravel()

    def iteration(c):
        x_d = c * x_step
        y_d = c * y_step
        i_d = c * y_step
        for ph in range(pooled_height):
            hstart = ph * stride_h - new_pads[0, 0]
            hend = hstart + kernel_shape[0] * dilation_h
            for pw in range(pooled_width):
                wstart = pw * stride_w - new_pads[1, 0]
                wend = wstart + kernel_shape[1] * dilation_w
                for pd in range(pooled_depth):
                    dstart = pd * stride_d - new_pads[2, 0]
                    dend = dstart + kernel_shape[2] * dilation_d
                    pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd
                    Yh = None
                    h_index = -1
                    w_index = -1
                    d_index = -1
                    for h in range(hstart, hend, dilation_h):
                        if h < 0 or h >= height:
                            continue
                        for w in range(wstart, wend, dilation_w):
                            if w < 0 or w >= width:
                                continue
                            for d in range(dstart, dend, dilation_d):
                                if d < 0 or d >= depth:
                                    continue
                                input_index = h * width * depth + w * depth + d
                                if Yh is None or X_data[x_d + input_index] > Yh:
                                    Yh = X_data[x_d + input_index]
                                    h_index = h
                                    w_index = w
                                    d_index = d
                    Y_data[y_d + pool_index] = Yh
                    I_data[i_d + pool_index] = c * x_step + h_index * width * depth + w_index * depth + d_index if storage_order == 0 else c * x_step + h_index + w_index * height + d_index * height * width
    for c in range(total_channels):
        iteration(c)
    if len(self.output) == 1:
        return (Y_data.reshape(y_dims),)
    return (Y_data.reshape(y_dims), I_data.reshape(y_dims))
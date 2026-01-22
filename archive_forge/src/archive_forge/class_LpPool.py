import numpy as np
from onnx.reference.ops.op_pool_common import CommonPool
class LpPool(CommonPool):

    def _run(self, x, auto_pad=None, ceil_mode=None, dilations=None, kernel_shape=None, p=2, pads=None, strides=None, count_include_pad=None):
        power_average = CommonPool._run(self, 'AVG', count_include_pad, np.power(np.absolute(x), p), auto_pad=auto_pad, ceil_mode=ceil_mode, dilations=dilations, kernel_shape=kernel_shape, pads=pads, strides=strides)
        kernel_element_count = np.prod(kernel_shape)
        return (np.power(kernel_element_count * power_average[0], 1.0 / p),)
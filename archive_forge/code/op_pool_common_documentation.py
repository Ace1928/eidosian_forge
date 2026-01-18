import itertools
import math
from typing import Sequence, Tuple, Union
import numpy as np
from onnx.reference.op_run import OpRun
This function is used to calculate the pooling result of a padded tensor
    padded: the padded tensor
    x_shape: the shape of the original tensor in [N, C, *spatial_shape]
    kernel: the pooling kernel
    strides: the strides
    out_shape: the shape of the output tensor
    pooling_type: the pooling type, can be "AVG", "LPPOOL", or "MAX"
    pads: the padding in an order of head_pad_1, head_pad_2, ..., tail_pad_1, tail_pad_2, ...
    dilations: the dilation
    count_include_pad: whether to include the padding in the calculation of average and lp pooling
    p: the p value for lp pooling
    
import itertools
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_indices import _get_index, _get_indices
def _get_output_shape_no_ceil(auto_pad: str, input_spatial_shape: Tuple[int], kernel_spatial_shape: Tuple[int], strides_spatial: Tuple[int]) -> Tuple[int]:
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(np.ceil(float(input_spatial_shape[i]) / float(strides_spatial[i])))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(np.ceil(float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / float(strides_spatial[i])))
    return tuple(out_shape)
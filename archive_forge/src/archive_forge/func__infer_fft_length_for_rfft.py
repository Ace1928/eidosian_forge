import re
import numpy as np
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import array_ops_stack as _array_ops_stack
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _infer_fft_length_for_rfft(input_tensor, fft_rank):
    """Infers the `fft_length` argument for a `rank` RFFT from `input_tensor`."""
    fft_shape = input_tensor.get_shape()[-fft_rank:]
    if not fft_shape.is_fully_defined():
        return _array_ops.shape(input_tensor)[-fft_rank:]
    return _ops.convert_to_tensor(fft_shape.as_list(), _dtypes.int32)
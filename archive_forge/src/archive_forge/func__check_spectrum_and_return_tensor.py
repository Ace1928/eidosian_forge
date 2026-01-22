import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
def _check_spectrum_and_return_tensor(self, spectrum):
    """Static check of spectrum.  Then return `Tensor` version."""
    spectrum = linear_operator_util.convert_nonref_to_tensor(spectrum, name='spectrum')
    if spectrum.shape.ndims is not None:
        if spectrum.shape.ndims < self.block_depth:
            raise ValueError(f'Argument `spectrum` must have at least {self.block_depth} dimensions. Received: {spectrum}.')
    return spectrum
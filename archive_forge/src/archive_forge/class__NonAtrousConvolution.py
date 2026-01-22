import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
class _NonAtrousConvolution:
    """Helper class for _non_atrous_convolution.

  Note that this class assumes that shapes of input and filter passed to
  `__call__` are compatible with `input_shape` and filter_shape passed to the
  constructor.

  Args:
    input_shape: static input shape, i.e. input.shape.
    filter_shape: static filter shape, i.e. filter.shape.
    padding: see _non_atrous_convolution.
    data_format: see _non_atrous_convolution.
    strides: see _non_atrous_convolution.
    name: see _non_atrous_convolution.
    num_batch_dims: (Optional.)  The number of batch dimensions in the input;
     if not provided, the default of `1` is used.
  """

    def __init__(self, input_shape, filter_shape, padding, data_format=None, strides=None, name=None, num_batch_dims=1):
        if input_shape.ndims is not None:
            filter_shape = filter_shape.with_rank(input_shape.ndims - num_batch_dims + 1)
        self.padding = padding
        self.name = name
        if filter_shape.ndims is not None:
            input_shape = input_shape.with_rank(filter_shape.ndims + num_batch_dims - 1)
        if input_shape.ndims is None:
            raise ValueError(f'Rank of convolution must be known. Received: input_shape={input_shape} of rank {input_shape.rank}')
        if input_shape.ndims < 3 or input_shape.ndims - num_batch_dims + 1 > 5:
            raise ValueError(f'`input_shape.rank - num_batch_dims + 1` must be at least 3 and at most 5. Received: input_shape.rank={input_shape.rank} and num_batch_dims={num_batch_dims}')
        conv_dims = input_shape.ndims - num_batch_dims - 1
        if strides is None:
            strides = [1] * conv_dims
        elif len(strides) != conv_dims:
            raise ValueError(f'`len(strides)` should be {conv_dims}. Received: strides={strides} of length {len(strides)}')
        if conv_dims == 1:
            if data_format is None:
                data_format = 'NWC'
            elif data_format not in {'NCW', 'NWC', 'NCHW', 'NHWC'}:
                raise ValueError(f"`data_format` must be 'NWC' or 'NCW'. Received: data_format={data_format}")
            self.strides = strides[0]
            self.data_format = data_format
            self.conv_op = self._conv1d
        elif conv_dims == 2:
            if data_format is None or data_format == 'NHWC':
                data_format = 'NHWC'
                strides = [1] + list(strides) + [1]
            elif data_format == 'NCHW':
                strides = [1, 1] + list(strides)
            else:
                raise ValueError(f"`data_format` must be 'NHWC' or 'NCHW'. Received: data_format={data_format}")
            self.strides = strides
            self.data_format = data_format
            self.conv_op = conv2d
        elif conv_dims == 3:
            if data_format is None or data_format == 'NDHWC':
                strides = [1] + list(strides) + [1]
            elif data_format == 'NCDHW':
                strides = [1, 1] + list(strides)
            else:
                raise ValueError(f"`data_format` must be 'NDHWC' or 'NCDHW'. Received: data_format={data_format}")
            self.strides = strides
            self.data_format = data_format
            self.conv_op = _conv3d_expanded_batch

    def _conv1d(self, input, filter, strides, padding, data_format, name):
        return conv1d(value=input, filters=filter, stride=strides, padding=padding, data_format=data_format, name=name)

    def __call__(self, inp, filter):
        return self.conv_op(input=inp, filter=filter, strides=self.strides, padding=self.padding, data_format=self.data_format, name=self.name)
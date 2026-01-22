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
class Convolution:
    """Helper class for convolution.

  Note that this class assumes that shapes of input and filter passed to
  `__call__` are compatible with `input_shape`, `filter_shape`, and
  `num_spatial_dims` passed to the constructor.

  Arguments
    input_shape: static shape of input. i.e. input.shape.  Its length is
      `batch_shape + input_spatial_shape + [num_channels]` if `data_format`
      does not start with `NC`, or
      `batch_shape + [num_channels] + input_spatial_shape` if `data_format`
      starts with `NC`.
    filter_shape: static shape of the filter. i.e. filter.shape.
    padding: The padding algorithm, must be "SAME" or "VALID".
    strides: see convolution.
    dilation_rate: see convolution.
    name: see convolution.
    data_format: A string or `None`.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (if `data_format` is `None`
      or does not start with `NC`), or the first post-batch dimension (i.e. if
      `data_format` starts with `NC`).
    num_spatial_dims: (Usually optional.)  Python integer, the rank of the
      spatial and channel dimensions.  For `1-D`, `2-D` and `3-D` convolutions,
      the value of `num_spatial_dims` is `1`, `2`, and `3`, respectively.
      This argument is only required to disambiguate the rank of `batch_shape`
      when `filter_shape.ndims is None` and `len(batch_shape) > 1`.  For
      backwards compatibility, if `num_spatial_dims is None` and
      `filter_shape.ndims is None`, then `len(batch_shape)` is assumed to be
      `1` (i.e., the input is expected to be
      `[batch_size, num_channels] + input_spatial_shape`
      or `[batch_size] + input_spatial_shape + [num_channels]`.
  """

    def __init__(self, input_shape, filter_shape, padding, strides=None, dilation_rate=None, name=None, data_format=None, num_spatial_dims=None):
        """Helper function for convolution."""
        num_batch_dims = None
        filter_shape = tensor_shape.as_shape(filter_shape)
        input_shape = tensor_shape.as_shape(input_shape)
        if filter_shape.ndims is not None:
            if num_spatial_dims is not None and filter_shape.ndims != num_spatial_dims + 2:
                raise ValueError(f'`filters.shape.rank` must be `num_spatial_dims + 2`. Received: filters.shape={filter_shape} of rank {filter_shape.rank} and num_spatial_dims={num_spatial_dims}')
            else:
                num_spatial_dims = filter_shape.ndims - 2
        if input_shape.ndims is not None and num_spatial_dims is not None:
            num_batch_dims = input_shape.ndims - num_spatial_dims - 1
        if num_spatial_dims is None:
            num_spatial_dims = input_shape.ndims - 2
        elif input_shape.ndims is not None:
            if input_shape.ndims < num_spatial_dims + 2:
                raise ValueError(f'`input.shape.rank` must be >= than `num_spatial_dims + 2`. Received: input.shape={input_shape} of rank {input_shape.rank} and num_spatial_dims={num_spatial_dims}')
            elif num_batch_dims is None:
                num_batch_dims = input_shape.ndims - num_spatial_dims - 1
        if num_spatial_dims is None:
            raise ValueError(f'When `num_spatial_dims` is not set, one of `input.shape.rank` or `filters.shape.rank` must be known. Received: input.shape={input_shape} of rank {input_shape.rank} and `filters.shape={filter_shape}` of rank {filter_shape.rank}')
        if num_batch_dims is None:
            num_batch_dims = 1
        if num_batch_dims < 1:
            raise ValueError(f'Batch dims should be >= 1, but found {num_batch_dims}. Batch dims was estimated as `input.shape.rank - num_spatial_dims - 1` and `num_spatial_dims` was either provided or estimated as `filters.shape.rank - 2`. Received: input.shape={input_shape} of rank {input_shape.rank}, filters.shape={filter_shape} of rank {filter_shape.rank}, and num_spatial_dims={num_spatial_dims}')
        if data_format is None or not data_format.startswith('NC'):
            input_channels_dim = tensor_shape.dimension_at_index(input_shape, num_spatial_dims + num_batch_dims)
            spatial_dims = range(num_batch_dims, num_spatial_dims + num_batch_dims)
        else:
            input_channels_dim = tensor_shape.dimension_at_index(input_shape, num_batch_dims)
            spatial_dims = range(num_batch_dims + 1, num_spatial_dims + num_batch_dims + 1)
        filter_dim = tensor_shape.dimension_at_index(filter_shape, num_spatial_dims)
        if not (input_channels_dim % filter_dim).is_compatible_with(0):
            raise ValueError(f'The number of input channels is not divisible by the corresponding number of output filters. Received: input.shape={input_shape} with {input_channels_dim} channels and filters.shape={filter_shape} with {filter_dim} output filters.')
        strides, dilation_rate = _get_strides_and_dilation_rate(num_spatial_dims, strides, dilation_rate)
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.data_format = data_format
        self.strides = strides
        self.padding = padding
        self.name = name
        self.dilation_rate = dilation_rate
        self.num_batch_dims = num_batch_dims
        self.num_spatial_dims = num_spatial_dims
        self.conv_op = _WithSpaceToBatch(input_shape, dilation_rate=dilation_rate, padding=padding, build_op=self._build_op, filter_shape=filter_shape, spatial_dims=spatial_dims, data_format=data_format, num_batch_dims=num_batch_dims)

    def _build_op(self, _, padding):
        return _NonAtrousConvolution(self.input_shape, filter_shape=self.filter_shape, padding=padding, data_format=self.data_format, strides=self.strides, name=self.name, num_batch_dims=self.num_batch_dims)

    def __call__(self, inp, filter):
        if device_context.enclosing_tpu_context() is not None:
            return convolution_internal(inp, filter, strides=self.strides, padding=self.padding, data_format=self.data_format, dilations=self.dilation_rate, name=self.name, call_from_convolution=False, num_spatial_dims=self.num_spatial_dims)
        else:
            return self.conv_op(inp, filter)
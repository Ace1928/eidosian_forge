import collections
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
def _random_out_op(self, in_shape, filter_shape, strides, padding, dilations):
    in_op = self._random_data_op(in_shape)
    filter_op = self._random_data_op(filter_shape)
    conv_op = nn_ops.conv2d(in_op, filter_op, strides=strides, padding=padding, dilations=dilations)
    out_shape = conv_op.get_shape()
    out_op = self._random_data_op(out_shape)
    return out_op
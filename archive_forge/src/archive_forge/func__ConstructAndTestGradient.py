import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
def _ConstructAndTestGradient(self, input_shape, filter_shape, output_shape, stride, padding, data_type, test_input, use_gpu, grouped_conv=False, data_format='NHWC', dilations=None):
    input_size = 1
    for x in input_shape:
        input_size *= x
    filter_size = 1
    for x in filter_shape:
        filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    input_np = np.array(input_data).reshape(input_shape)
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
    filter_np = np.array(filter_data).reshape(filter_shape)
    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with self.session(graph=graph, use_gpu=use_gpu) as sess:
        tolerance = {dtypes.float16: 4.0, dtypes.float32: 0.0008, dtypes.float64: 1e-12, dtypes.bfloat16: 1.0}[data_type]
        input_tensor = constant_op.constant(input_np, shape=input_shape, dtype=data_type, name='input')
        filter_tensor = constant_op.constant(filter_np, shape=filter_shape, dtype=data_type, name='filter')
        native_input = input_tensor
        strides = [1, stride, stride, 1]
        if isinstance(padding, list):
            padding = [(0, 0)] + padding + [(0, 0)]
        if data_format == 'NCHW':
            native_input = array_ops.transpose(input_tensor, [0, 3, 1, 2])
            input_shape = [input_shape[0], input_shape[3], input_shape[1], input_shape[2]]
            output_shape = [output_shape[0], output_shape[3], output_shape[1], output_shape[2]]
            strides = [1, 1, stride, stride]
            if isinstance(padding, list):
                padding = [padding[0], padding[3], padding[1], padding[2]]
        with sess.graph._kernel_label_map({'DepthwiseConv2dNative': 'cudnn_grouped_convolution', 'DepthwiseConv2dNativeBackpropInput': 'cudnn_grouped_convolution', 'DepthwiseConv2dNativeBackpropFilter': 'cudnn_grouped_convolution'} if grouped_conv else {}):
            depthwise_conv2d = nn_impl.depthwise_conv2d(native_input, filter_tensor, strides, padding, data_format=data_format, dilations=dilations, name='depthwise_conv2d')
        self.assertEqual(output_shape, depthwise_conv2d.get_shape())
        try:
            if test_input:
                err = gradient_checker.compute_gradient_error(native_input, input_shape, depthwise_conv2d, output_shape)
            else:
                err = gradient_checker.compute_gradient_error(filter_tensor, filter_shape, depthwise_conv2d, output_shape)
        except errors.InvalidArgumentError as e:
            if grouped_conv and "No OpKernel was registered to support Op 'DepthwiseConv2dNative'" in e.message:
                tf_logging.warn('Skipping grouped convolution test')
                return
            raise e
        tf_logging.info('data_type: %r, use_gpu: %r, grouped_conv: %r, error = %f', data_type, use_gpu, grouped_conv, err)
        self.assertLess(err, tolerance)
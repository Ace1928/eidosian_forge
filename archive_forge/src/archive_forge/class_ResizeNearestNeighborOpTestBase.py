from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
@test_util.for_all_test_methods(test_util.disable_xla, 'align_corners=False not supported by XLA')
class ResizeNearestNeighborOpTestBase(test.TestCase):
    TYPES = [np.float16, np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]

    def testShapeIsCorrectAfterOp(self):
        in_shape = [1, 2, 2, 1]
        out_shape = [1, 4, 6, 1]
        for nptype in self.TYPES:
            x = np.arange(0, 4).reshape(in_shape).astype(nptype)
            input_tensor = constant_op.constant(x, shape=in_shape)
            resize_out = image_ops.resize_nearest_neighbor(input_tensor, out_shape[1:3])
            with self.cached_session():
                self.assertEqual(out_shape, list(resize_out.get_shape()))
                resize_out = self.evaluate(resize_out)
            self.assertEqual(out_shape, list(resize_out.shape))

    def testGradFromResizeToLargerInBothDims(self):
        in_shape = [1, 2, 3, 1]
        out_shape = (1, 4, 6, 1)
        for nptype in self.TYPES:
            x = np.arange(0, 6).reshape(in_shape).astype(nptype)

            def resize_nn(t, shape=out_shape):
                return image_ops.resize_nearest_neighbor(t, shape[1:3])
            with self.cached_session():
                input_tensor = constant_op.constant(x, shape=in_shape)
                err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(resize_nn, [input_tensor], delta=1 / 8))
                self.assertLess(err, 0.001)

    def testGradFromResizeToSmallerInBothDims(self):
        in_shape = [1, 4, 6, 1]
        out_shape = (1, 2, 3, 1)
        for nptype in self.TYPES:
            x = np.arange(0, 24).reshape(in_shape).astype(nptype)

            def resize_nn(t, shape=out_shape):
                return image_ops.resize_nearest_neighbor(t, shape[1:3])
            with self.cached_session():
                input_tensor = constant_op.constant(x, shape=in_shape)
                err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(resize_nn, [input_tensor], delta=1 / 8))
                self.assertLess(err, 0.001)

    def testCompareGpuVsCpu(self):
        in_shape = [1, 4, 6, 3]
        out_shape = (1, 8, 16, 3)
        for nptype in self.TYPES:
            x = np.arange(0, np.prod(in_shape)).reshape(in_shape).astype(nptype)
            for align_corners in [True, False]:

                def resize_nn(t, shape=out_shape, align_corners=align_corners):
                    return image_ops.resize_nearest_neighbor(t, shape[1:3], align_corners=align_corners)
                with self.cached_session(use_gpu=False):
                    input_tensor = constant_op.constant(x, shape=in_shape)
                    grad_cpu = gradient_checker_v2.compute_gradient(resize_nn, [input_tensor], delta=1 / 8)
                with self.cached_session():
                    input_tensor = constant_op.constant(x, shape=in_shape)
                    grad_gpu = gradient_checker_v2.compute_gradient(resize_nn, [input_tensor], delta=1 / 8)
                self.assertAllClose(grad_cpu, grad_gpu, rtol=1e-05, atol=1e-05)
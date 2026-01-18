from tensorflow.python.framework import test_util as _test_util
from tensorflow.python.platform import googletest as _googletest
from tensorflow.python.framework.test_util import assert_equal_graph_def
from tensorflow.python.framework.test_util import create_local_cluster
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.framework.test_util import gpu_device_name
from tensorflow.python.framework.test_util import is_gpu_available
from tensorflow.python.ops.gradient_checker import compute_gradient_error
from tensorflow.python.ops.gradient_checker import compute_gradient
import functools
import sys
from tensorflow.python.util.tf_export import tf_export
@functools.wraps(func)
def wrapper_disable_with_predicate(self, *args, **kwargs):
    if pred():
        self.skipTest(skip_message)
    else:
        return func(self, *args, **kwargs)
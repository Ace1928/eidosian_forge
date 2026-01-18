from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from concurrent import futures
import threading
import time
import uuid
from absl.testing import parameterized
import numpy as np
from seed_rl.grpc.python import ops
from six.moves import range
import tensorflow as tf
def test_tpu_tf_function_same_device(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    with tf.device('/device:CPU:0'):
        a = tf.Variable(1)
    with tf.device('/device:CPU:0'):

        @tf.function
        def get_a_plus_one():
            return a + 1

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
        with tf.device('/device:CPU:0'):
            b = x + get_a_plus_one()
        return b + 1
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    a = client.foo(1)
    self.assertAllEqual(4, a)
    server.shutdown()
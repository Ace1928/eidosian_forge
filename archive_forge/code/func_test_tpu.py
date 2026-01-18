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
def test_tpu(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    with tf.device('/device:CPU:0'):
        a = tf.Variable(1)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
        with tf.device('/device:CPU:0'):
            b = a + 1
            c = x + 1
        return (x + b, c)
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    a, b = client.foo(42)
    self.assertAllEqual(44, a)
    self.assertAllEqual(43, b)
    server.shutdown()
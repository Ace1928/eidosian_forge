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
def test_variable_out_of_scope(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    def bind():
        a = tf.Variable(1)

        @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
        def foo(x):
            return x + a
        server.bind(foo)
    bind()
    server.start()
    client = ops.Client(address)
    self.assertAllEqual(43, client.foo(42))
    server.shutdown()
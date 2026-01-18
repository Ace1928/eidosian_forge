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
def test_wait_for_server2(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
        return x + 1
    server.bind(foo)

    def create_and_send():
        client = ops.Client(address)
        self.assertAllEqual(43, client.foo(42))
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(create_and_send)
        time.sleep(2)
        server.start()
        f.result()
        server.shutdown()
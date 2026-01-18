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
def test_stress_test(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([5], tf.int32)])
    def foo(x):
        return x + 1
    server.bind(foo)
    server.start()
    num_clients = 10
    num_calls = 100
    clients = [ops.Client(address) for _ in range(num_clients)]

    def do_calls(client):
        for i in range(num_calls):
            self.assertAllEqual(i + 1, client.foo(i))
    with futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
        fs = [executor.submit(do_calls, client) for client in clients]
        for i, f in enumerate(futures.as_completed(fs), 0):
            f.result()
            if i == num_clients // 2:
                try:
                    server.shutdown()
                except tf.errors.UnavailableError:
                    pass
                break
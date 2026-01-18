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
def test_shutdown_while_in_call(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    is_waiting = threading.Event()

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
        tf.py_function(is_waiting.set, [], [])
        tf.py_function(time.sleep, [1], [])
        return x + 1
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(client.foo, 42)
        is_waiting.wait()
        server.shutdown()
        with self.assertRaisesRegex(tf.errors.UnavailableError, 'server closed'):
            f.result()
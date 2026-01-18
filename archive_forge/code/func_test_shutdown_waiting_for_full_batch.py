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
def test_shutdown_waiting_for_full_batch(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)])
    def foo(x):
        return x + 1
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(client.foo, 42)
        time.sleep(1)
        server.shutdown()
        with self.assertRaisesRegex(tf.errors.UnavailableError, 'server closed'):
            f.result()
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
def test_not_bound(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    with self.assertRaisesRegex(tf.errors.UnavailableError, 'No function was bound'):
        server.start()
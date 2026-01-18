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
def test_server_non_vector_server_addresses(self):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'server_addresses must be a vector'):
        ops.Server([['localhost:8000', 'localhost:8001']])
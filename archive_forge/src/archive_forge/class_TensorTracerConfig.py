import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
class TensorTracerConfig(object):
    """Tensor Tracer config object."""

    def __init__(self):
        self.version = _CURRENT_VERSION
        self.device_type = None
        self.num_replicas = None
        self.num_replicas_per_host = None
        self.num_hosts = None
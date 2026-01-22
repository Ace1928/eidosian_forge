import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _produce_tensor(self, name, timestamp, tensors_pid, allocator, num_bytes):
    object_id = len(self._tensors)
    tensor = _TensorTracker(name, object_id, timestamp, tensors_pid, allocator, num_bytes)
    self._tensors[name] = tensor
    return tensor
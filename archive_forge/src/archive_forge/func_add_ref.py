import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def add_ref(self, timestamp):
    """Adds a reference to this tensor with the specified timestamp.

    Args:
      timestamp:  Timestamp of object reference as an integer.
    """
    self._ref_times.append(timestamp)
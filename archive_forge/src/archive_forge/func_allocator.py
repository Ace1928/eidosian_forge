import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
@property
def allocator(self):
    """Name of the allocator used to create this tensor (string)."""
    return self._allocator
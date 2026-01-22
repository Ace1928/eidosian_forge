import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _is_gputrace_device(self, device_name):
    """Returns true if this device is part of the GPUTracer logging."""
    return '/stream:' in device_name or '/memcpy' in device_name
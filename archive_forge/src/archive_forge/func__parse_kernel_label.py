import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _parse_kernel_label(self, label, node_name):
    """Parses the fields in a node timeline label."""
    start = label.find('@@')
    end = label.find('#')
    if start >= 0 and end >= 0 and (start + 2 < end):
        node_name = label[start + 2:end]
    fields = node_name.split(':') + ['unknown']
    name, op = fields[:2]
    return (name, op)
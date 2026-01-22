import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _load_log_message_from_event_file(event_file_path):
    event = event_pb2.Event()
    with gfile.Open(event_file_path, 'rb') as f:
        event.ParseFromString(f.read())
    return event.log_message.message
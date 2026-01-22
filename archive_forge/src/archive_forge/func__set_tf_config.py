import collections
import contextlib
import json
import os
import signal
import sys
import threading
import time
import unittest
import weakref
from absl import logging
import six
from six.moves import queue as Queue
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.util.tf_export import tf_export
def _set_tf_config(task_type, task_id, cluster_spec, rpc_layer=None):
    """Set TF_CONFIG environment variable."""
    tf_config_dict = {'cluster': cluster_spec, 'task': {'type': task_type, 'index': task_id}}
    if rpc_layer is not None:
        tf_config_dict['rpc_layer'] = rpc_layer
    os.environ['TF_CONFIG'] = json.dumps(tf_config_dict)
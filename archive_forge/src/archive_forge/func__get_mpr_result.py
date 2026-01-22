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
def _get_mpr_result(self, process_statuses):
    stdout = self._queue_to_list(self._streaming_queue)
    return_values = []
    for process_status in process_statuses.values():
        if process_status.return_value is not None:
            return_values.append(process_status.return_value)
    return MultiProcessRunnerResult(stdout=stdout, return_value=return_values)
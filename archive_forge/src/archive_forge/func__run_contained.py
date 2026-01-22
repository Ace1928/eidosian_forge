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
def _run_contained(task_type, task_id, fn, args, kwargs):
    """Runs `fn` with `args` and `kwargs`.

  The function returns _ProcessStatusInfo which captures the return value and
  the exception.

  Args:
    task_type: the task type.
    task_id: the task index.
    fn: the function to be run.
    args: optional positional arguments to be supplied in `fn`.
    kwargs: optional keyword arguments to be supplied in `fn`.

  Returns:
    a _ProcessStatusInfo.

  """
    is_successful = False
    return_value = None
    exc_info = None
    try:
        return_value = fn(*args, **kwargs)
        is_successful = True
        return _ProcessStatusInfo(task_type=task_type, task_id=task_id, is_successful=is_successful, exc_info=exc_info, return_value=return_value)
    except Exception:
        exc_info = sys.exc_info()
        return _ProcessStatusInfo(task_type=task_type, task_id=task_id, is_successful=is_successful, exc_info=exc_info, return_value=return_value)
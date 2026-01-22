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
def _pool_runner_worker(task_type, task_id, initializer, conn):
    """Function that runs on the workers in a pool.

  It listens for callables to run and returns the result until `conn` is closed.
  It captures the exceptions during executing the callable and return it through
  `conn`.

  Args:
    task_type: the task type.
    task_id: the task index.
    initializer: a callable to execute during startup.
    conn: a multiprocessing.Connection object to listen for tasks and send
      results.
  """
    if initializer:
        initializer = dill.loads(initializer)
        initializer()
    while True:
        try:
            fn, args, kwargs = conn.recv()
        except EOFError:
            break
        fn = dill.loads(fn)
        info = _run_contained(task_type, task_id, fn, args, kwargs)
        sys.stdout.flush()
        sys.stderr.flush()
        conn.send(info)
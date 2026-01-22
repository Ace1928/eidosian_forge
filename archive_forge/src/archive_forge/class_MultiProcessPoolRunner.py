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
class MultiProcessPoolRunner(object):
    """A utility class to start a process pool to simulate a cluster.

  It's similar to MultiProcessRunner, but uses a pool of processes to avoid the
  expensive initialization cost of Tensorflow.
  """

    def __init__(self, cluster_spec, initializer=None, share_gpu=True):
        """Creates a multi-process pool runner.

    Args:
      cluster_spec: Dict for cluster spec. The following is an example of
        cluster with three workers.
        {"worker": ["worker0.example.com:2222",
                    "worker1.example.com:2222",
                    "worker2.example.com:2222"]}
      initializer: a callable to called at the startup of worker processes.
      share_gpu: Whether to share GPUs among workers. If False, each worker is
        assigned different GPUs in a roundrobin fashion.

    Raises:
      RuntimeError: if `multi_process_runner.test_main()` is not called.
      ValueError: if there are more than one chief in the `cluster_spec`.
    """
        _active_pool_runners.add(self)
        self._cluster_spec = cluster_spec
        self._initializer = initializer
        self._share_gpu = share_gpu
        self._conn = {}
        self._runner = None

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shuts down the worker pool."""
        for conn in self._conn.values():
            conn.close()
        self._conn = {}
        if self._runner is not None:
            try:
                self._runner.join()
            except Exception as e:
                logging.error('Ignoring exception when shutting down MultiProcessPoolRunner: %s', e)
            self._runner = None

    def _start(self):
        """Starts the worker pool."""
        if dill is None:
            raise unittest.SkipTest('TODO(b/150264776): Resolve dependency issue in CI')
        self._runner = MultiProcessRunner(fn=lambda: None, cluster_spec=self._cluster_spec, use_dill_for_args=False, share_gpu=self._share_gpu)
        if self._initializer:
            initializer = dill.dumps(self._initializer, dill.HIGHEST_PROTOCOL)
        else:
            initializer = None
        for task_type, addresses in self._cluster_spec.items():
            for task_id, _ in enumerate(addresses):
                conn1, conn2 = multiprocessing.Pipe(duplex=True)
                self._conn[task_type, task_id] = conn1
                self._runner.start_single_process(task_type, task_id, fn=_pool_runner_worker, args=(task_type, task_id, initializer, conn2))

    def run(self, fn, args=None, kwargs=None):
        """Runs `fn` with `args` and `kwargs` on all jobs.

    Args:
      fn: The function to be run.
      args: Optional positional arguments to be supplied in `fn`.
      kwargs: Optional keyword arguments to be supplied in `fn`.

    Returns:
      A list of return values.
    """
        _check_initialization()
        multi_process_lib.Process()
        if self._runner is None:
            self._start()
        fn = dill.dumps(fn, dill.HIGHEST_PROTOCOL)
        for conn in self._conn.values():
            conn.send((fn, args or [], kwargs or {}))
        process_statuses = []
        for (task_type, task_id), conn in self._conn.items():
            logging.info('Waiting for the result from %s-%d', task_type, task_id)
            try:
                process_statuses.append(conn.recv())
            except EOFError:
                self.shutdown()
                raise RuntimeError('Unexpected EOF. Worker process may have died. Please report a bug')
        return_values = []
        for process_status in process_statuses:
            assert isinstance(process_status, _ProcessStatusInfo)
            if not process_status.is_successful:
                six.reraise(*process_status.exc_info)
            if process_status.return_value is not None:
                return_values.append(process_status.return_value)
        return return_values
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
def _start_subprocess_and_reading_thread(self, task_type, task_id, cluster_spec=None, fn=None, args=None, kwargs=None):
    """Start a subprocess and a thread the reads lines from the subprocess."""
    if dill is None:
        raise unittest.SkipTest('TODO(b/150264776): Resolve dependency issue in CI')
    cluster_spec = cluster_spec or self._cluster_spec
    visible_gpus = None
    if not self._share_gpu and self._total_gpu > 0:
        id_in_cluster = multi_worker_util.id_in_cluster(cluster_spec, task_type, task_id)
        worker_count = multi_worker_util.worker_count(cluster_spec, task_type)
        visible_gpus = list(range(id_in_cluster, self._total_gpu, worker_count))
    test_env = TestEnvironment(task_type=task_type, task_id=task_id, cluster_spec=cluster_spec, rpc_layer=self._rpc_layer, grpc_fail_fast=self._grpc_fail_fast, v2_enabled=self._v2_enabled, executing_eagerly=self._executing_eagerly, visible_gpus=visible_gpus)
    pipe_r, pipe_w = multiprocessing.Pipe(duplex=False)
    resources = Resources(process_status_queue=self._process_status_queue, parent_to_sub_queue=self._parent_to_sub_queue, streaming_pipe_w=pipe_w, barrier=self._barrier)
    if fn is None:
        fn, args, kwargs = (self._fn, self._args, self._kwargs)
    fn = dill.dumps(fn, dill.HIGHEST_PROTOCOL)
    if self._use_dill_for_args:
        args = dill.dumps(args, dill.HIGHEST_PROTOCOL)
        kwargs = dill.dumps(kwargs, dill.HIGHEST_PROTOCOL)
    p = _Process(test_env=test_env, target=_ProcFunc(), args=(resources, test_env, fn, args, kwargs, self._use_dill_for_args), daemon=self._daemon)
    p.start()
    self._processes[task_type, task_id] = p
    self._terminated.discard((task_type, task_id))
    thread = threading.Thread(target=self._continuously_readline_from_sub, args=(pipe_r, task_type, task_id))
    thread.start()
    self._reading_threads.append(thread)
    if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
        self._watchdog_thread = threading.Thread(target=self._process_watchdog)
        self._watchdog_thread.start()
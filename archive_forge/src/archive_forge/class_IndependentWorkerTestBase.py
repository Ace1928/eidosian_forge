import contextlib
import copy
import json
import os
import subprocess
import sys
import threading
import unittest
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class IndependentWorkerTestBase(test.TestCase):
    """Testing infra for independent workers."""

    def _make_mock_run_std_server(self):

        def _mock_run_std_server(*args, **kwargs):
            """Returns the std server once all threads have started it."""
            with skip_if_grpc_server_cant_be_started(self):
                ret = original_run_std_server(*args, **kwargs)
            if not getattr(self._thread_local, 'server_started', False):
                self._barrier.wait()
            self._thread_local.server_started = True
            return ret
        return _mock_run_std_server

    def setUp(self):
        self._mock_os_env = MockOsEnv()
        self._mock_context = test.mock.patch.object(os, 'environ', self._mock_os_env)
        self._coord = coordinator.Coordinator()
        super(IndependentWorkerTestBase, self).setUp()
        self._mock_context.__enter__()
        self._thread_local = threading.local()

    def tearDown(self):
        self._mock_context.__exit__(None, None, None)
        super(IndependentWorkerTestBase, self).tearDown()

    def _task_thread(self, task_fn, tf_config, executing_eagerly, *args, **kwargs):
        with self._coord.stop_on_exception():
            os.environ['TF_CONFIG'] = json.dumps(tf_config)
            if executing_eagerly:
                with context.eager_mode():
                    task_fn(*args, **kwargs)
            else:
                with ops.Graph().as_default(), context.graph_mode():
                    task_fn(*args, **kwargs)

    def _run_task_in_thread(self, task_fn, cluster_spec, task_type, task_id, *args, **kwargs):
        """Run tasks in a thread.

    If `tf_config` is provided, use it for the new thread; if not, construct one
    from `cluster_spec`, `task_type`, and `task_id`, and provide it to the new
    thread to be set as `TF_CONFIG` environment.

    Args:
      task_fn: The function to run in the new thread.
      cluster_spec: The cluster spec.
      task_type: The task type.
      task_id: The task id.
      *args: Additional positional arguments to provide to the thread's task_fn.
      **kwargs: Additional keyword arguments to provide to the thread's task_fn.
        If `tf_config` is provided, that dict will be used for the TF_CONFIG for
        the new thread.

    Returns:
      The thread that has started.
    """
        tf_config = kwargs.pop('tf_config', None)
        if tf_config is None:
            if task_type:
                tf_config = {'cluster': cluster_spec, 'task': {'type': task_type, 'index': task_id}}
            else:
                tf_config = {'cluster': cluster_spec}
        t = threading.Thread(target=self._task_thread, args=(task_fn, tf_config, context.executing_eagerly()) + args, kwargs=kwargs)
        t.start()
        return t

    def run_multiple_tasks_in_threads(self, task_fn, cluster_spec, *args, **kwargs):
        threads = {}
        for task_type in cluster_spec.keys():
            threads[task_type] = []
            for task_id in range(len(cluster_spec[task_type])):
                t = self._run_task_in_thread(task_fn, cluster_spec, task_type, task_id, *args, **kwargs)
                threads[task_type].append(t)
        return threads

    def join_independent_workers(self, worker_threads):
        with skip_if_grpc_server_cant_be_started(self):
            self._coord.join(worker_threads)
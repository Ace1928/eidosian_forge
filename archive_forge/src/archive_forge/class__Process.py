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
class _Process(multi_process_lib.Process):
    """A modified `multiprocessing.Process` that can set up environment variables."""

    def __init__(self, test_env, **kwargs):
        super(_Process, self).__init__(**kwargs)
        self._test_env = test_env
        self._actual_run = getattr(self, 'run')
        self.run = self._run_with_setenv

    def _run_with_setenv(self):
        test_env = self._test_env
        if test_env.grpc_fail_fast is not None:
            os.environ['GRPC_FAIL_FAST'] = str(test_env.grpc_fail_fast)
        if test_env.visible_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in test_env.visible_gpus])
        _set_tf_config(test_env.task_type, test_env.task_id, test_env.cluster_spec, test_env.rpc_layer)
        return self._actual_run()
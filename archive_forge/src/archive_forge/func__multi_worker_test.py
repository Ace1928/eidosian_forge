import collections
import copy
import re
import sys
import types
import unittest
from absl import app
import six
from tensorflow.python.client import session
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations as framework_combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _multi_worker_test(test_method):
    """Decorate test_method so that it runs in each worker.

  We use `multi_process_runner` to simulate multiple workers. Since we run the
  this function in the main process and all worker processes, this decoration
  behaves differently in the main process and worker procssses. In the main
  process, it spawns subprocesses and runs the test on each of them; in a worker
  process, it executes test in the same way as a normal test, e.g.
  setUp()/tearDown() are called before/after the test.

  Args:
    test_method: a function which must be a test method.

  Returns:
    Decorated `test_method`. Note that the decorated function has additional
    arguments.
  """

    def decorator(self, has_chief, num_workers, num_ps, share_gpu, runner, **kwargs):
        if _num_total_workers(has_chief, num_workers) == 1 or _running_in_worker or (test_util.is_xla_enabled() and num_ps > 0):
            with _multi_worker_session(kwargs):
                test_method(self, **kwargs)
            return
        test_id = self.id()
        if runner:
            results = runner.run(_test_runner, args=(test_id, _env))
        else:
            cluster_spec = multi_worker_test_base.create_cluster_spec(has_chief=has_chief, num_workers=num_workers, num_ps=num_ps, has_eval=False)
            ephemeral_runner = multi_process_runner.MultiProcessRunner(_test_runner, cluster_spec, share_gpu=share_gpu, args=(test_id, _env), dependence_on_chief=has_chief)
            ephemeral_runner.start()
            results = ephemeral_runner.join().return_value
        skip_reason = None
        for result in results:
            if result.status == 'failure':
                self.fail(result.message)
                break
            elif result.status == 'skipped':
                skip_reason = result.message
        if skip_reason is not None:
            self.skipTest(skip_reason)
    argspec = tf_inspect.getfullargspec(test_method)
    decorator_args = (argspec.args or []) + ['has_chief', 'num_workers', 'num_ps', 'share_gpu', 'runner']
    decorator_argspec = argspec._replace(args=decorator_args)
    return tf_decorator.make_decorator(test_method, decorator, decorator_argspec=decorator_argspec)
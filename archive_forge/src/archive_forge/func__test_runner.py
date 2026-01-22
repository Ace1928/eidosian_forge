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
def _test_runner(test_id, test_env):
    """Executes the test with the given test_id.

  This is a simple wrapper around TestRunner to be used with
  multi_process_runner. Similar to test.main(), but it executes only one test
  specified by test_id and returns whether the test succeeds. If the test fails,
  the function prints failures and errors to stdout.

  Args:
    test_id: TestCase.id()
    test_env: a TestEnvironment object.

  Returns:
    A boolean indicates whether the test succeeds.
  """
    global _running_in_worker, _env
    _running_in_worker = True
    _env = test_env
    test = unittest.defaultTestLoader.loadTestsFromName(test_id)
    runner = unittest.TextTestRunner()
    result = runner.run(test)
    failures = result.failures + result.expectedFailures + result.errors
    if failures:
        ret = _TestResult(status='failure', message=failures[0][1])
    elif result.skipped:
        ret = _TestResult(status='skipped', message=result.skipped[0][1])
    else:
        ret = _TestResult(status='ok', message=None)
    if ret.message:
        print(ret.message)
    return ret
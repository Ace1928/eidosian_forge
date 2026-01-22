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
class TPUCombination(combinations_lib.TestCombination):
    """Allow to request TPU hardware and skip non-TPU combinations.

  This class expects test_combinations to be generated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_tpus` parameter is supported.  TPU hardware is
  required, if its argument is `True` or > 0.

  Optionally, the `use_cloud_tpu` parameter is supported. If TPU hardware is
  required by `required_tpus`, it specifically must be a Cloud TPU (specified
  with `--tpu`) if `use_cloud_tpu` is `True`.

  Attributes:
    TPU_TEST: The environment is considered to have TPU hardware available if
              the name of the program contains "test_tpu".
  """
    TPU_TEST = False
    if sys.argv:
        TPU_TEST = 'test_tpu' in sys.argv[0]

    def should_execute_combination(self, kwargs):
        distributions = [v for v in kwargs.values() if isinstance(v, NamedDistribution)]
        if 'required_tpus' in kwargs and 'required_tpu' in kwargs:
            raise ValueError('Do not use `required_tpu`.  Both `required_tpus` and `required_tpu` were specified.')
        required_tpus = kwargs.get('required_tpus', None) or kwargs.get('required_tpu', None)
        if distributions and required_tpus:
            raise ValueError('Do not use `required_tpus` and arguments of type NamedDistribution together.')
        number_of_required_tpus = max([required_tpus or 0] + [d.required_tpu or 0 for d in distributions])
        use_cloud_tpu = any([kwargs.get('use_cloud_tpu')] + [d.use_cloud_tpu for d in distributions])
        tpu = hasattr(flags.FLAGS, 'tpu') and flags.FLAGS.tpu or ''
        if not number_of_required_tpus and TPUCombination.TPU_TEST:
            return (False, "Test that doesn't require TPUs.")
        if number_of_required_tpus and (not TPUCombination.TPU_TEST):
            return (False, "Test requires a TPU, but it's not available.")
        if use_cloud_tpu and (not tpu):
            return (False, 'Test requires a Cloud TPU, but none specified.')
        if not use_cloud_tpu and tpu:
            return (False, 'Test requires local TPU, but Cloud TPU specified.')
        return (True, None)

    def parameter_modifiers(self):
        return [combinations_lib.OptionalParameter('required_tpus'), combinations_lib.OptionalParameter('required_tpu'), combinations_lib.OptionalParameter('use_cloud_tpu')]
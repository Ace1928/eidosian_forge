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
class GPUCombination(combinations_lib.TestCombination):
    """Enable tests to request GPU hardware and skip non-GPU combinations.

  This class expects test_combinations to be generated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_gpus` argument is supported.  GPU hardware is
  required, if its value is `True` or > 0.

  Attributes:
    GPU_TEST: The environment is considered to have GPU hardware available if
              the name of the program contains "test_gpu" or "test_xla_gpu".
  """
    GPU_TEST = False
    if sys.argv:
        GPU_TEST = re.search('(test_2?gpu|test_xla_2?gpu)$', sys.argv[0])

    def should_execute_combination(self, kwargs):
        distributions = [v for v in kwargs.values() if isinstance(v, NamedDistribution)]
        required_gpus = kwargs.get('required_gpus', 0)
        required_physical_gpus = kwargs.get('required_physical_gpus', 0)
        if distributions and required_gpus:
            raise ValueError('Do not use `required_gpus` and arguments of type NamedDistribution together.')
        number_of_required_gpus = max([required_gpus] + [required_physical_gpus] + [d.required_physical_gpus or 0 for d in distributions] + [d.required_gpus or 0 for d in distributions])
        number_of_required_physical_gpus = max([required_physical_gpus] + [d.required_physical_gpus or 0 for d in distributions])
        if required_physical_gpus and required_gpus:
            raise ValueError('Only one of `required_physical_gpus`(number of physical GPUs required) and `required_gpus`(total number of GPUs required) should be set. ')
        if not number_of_required_gpus and GPUCombination.GPU_TEST:
            return (False, "Test that doesn't require GPUs.")
        elif number_of_required_gpus > 0 and context.num_gpus() < number_of_required_gpus:
            return (False, 'Only {} of {} required GPUs are available.'.format(context.num_gpus(), number_of_required_gpus))
        elif number_of_required_physical_gpus > len(config.list_physical_devices('GPU')):
            return (False, 'Only {} of {} required physical GPUs are available.'.format(config.list_physical_devices('GPU'), required_physical_gpus))
        else:
            return (True, None)

    def parameter_modifiers(self):
        return [combinations_lib.OptionalParameter('required_gpus'), combinations_lib.OptionalParameter('required_physical_gpus')]
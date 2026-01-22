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
class DistributionCombination(combinations_lib.TestCombination):
    """Sets up distribution strategy for tests."""

    def should_execute_combination(self, kwargs):
        distributions = [v for v in kwargs.values() if isinstance(v, NamedDistribution)]
        if test_util.is_xla_enabled() and any((d.no_xla for d in distributions)):
            return (False, 'n/a: skipping strategy combination with no_xla=True in XLA tests')
        return (True, None)

    def parameter_modifiers(self):
        return [DistributionParameter(), combinations_lib.OptionalParameter('use_var_policy')]
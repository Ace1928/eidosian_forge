from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
Returns a list of test case args that covers ops and test_configs.

  The list is a Cartesian product between op_list and test_configs.

  Args:
    op_list: A list of dicts, with items keyed by 'testcase_name' and 'op'.
      Available lists are defined later in this module.
    test_configs: A list of dicts, additional kwargs to be appended for each
      test parameters.

  Returns:
    test_configurations: a list of test parameters that covers all
      provided ops in op_list and args in test_configs.
  
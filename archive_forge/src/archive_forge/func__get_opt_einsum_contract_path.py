import collections
import functools
import re
import string
import numpy as np
import opt_einsum
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _get_opt_einsum_contract_path(equation, shaped_inputs_tuple, optimize):
    """Returns the (memoized) result of opt_einsum.contract_path."""
    _, contractions = opt_einsum.contract_path(equation, *shaped_inputs_tuple, optimize=optimize, einsum_call=True, use_blas=True)
    indices_and_equations = tuple([(expr[0], expr[2]) for expr in contractions])
    return indices_and_equations
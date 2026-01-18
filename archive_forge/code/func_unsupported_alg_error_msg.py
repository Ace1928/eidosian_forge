import enum
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
def unsupported_alg_error_msg(alg):
    """Produces the unsupported-algorithm error message."""
    if isinstance(alg, int):
        philox = Algorithm.PHILOX.value
        threefry = Algorithm.THREEFRY.value
        auto_select = Algorithm.AUTO_SELECT.value
    elif isinstance(alg, str):
        philox = 'philox'
        threefry = 'threefry'
        auto_select = 'auto_select'
    else:
        philox = Algorithm.PHILOX
        threefry = Algorithm.THREEFRY
        auto_select = Algorithm.AUTO_SELECT
    return f'Argument `alg` got unsupported value {alg}. Supported values are {philox} for the Philox algorithm, {threefry} for the ThreeFry algorithm, and {auto_select} for auto-selection.'
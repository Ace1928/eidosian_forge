import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
class _PythonLoopChecker(object):
    """Verifies Python loops for TF-specific limits."""
    __slots__ = ('iterations', 'check_inefficient_unroll', 'check_op_count_after_iteration', 'ops_before_iteration')

    def __init__(self):
        self.iterations = 1
        self.check_inefficient_unroll = WARN_INEFFICIENT_UNROLL
        self.check_op_count_after_iteration = False

    def _get_ops(self):
        return set(ops.get_default_graph().get_operations())

    def _check_unroll_limits(self):
        if self.iterations > PYTHON_MAX_ITERATIONS:
            raise ValueError('iteration limit exceeded')

    def _stop_checking_inefficient_unroll(self):
        self.check_inefficient_unroll = False
        self.check_op_count_after_iteration = False
        self.ops_before_iteration = None

    def _verify_inefficient_unroll(self):
        """Checks for possibly-inefficient creation of ops in a Python loop."""
        assert self.ops_before_iteration is not None
        ops_after_iteration = self._get_ops()
        new_ops = tuple((op for op in ops_after_iteration if op not in self.ops_before_iteration))
        if len(new_ops) < INEFFICIENT_UNROLL_MIN_OPS:
            return False
        ag_logging.warning('Large unrolled loop detected. Did you mean to use a TF loop? The following ops were created after iteration %s: %s\nSee https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/common_errors.md#warning-large-unrolled-loop-detected\nLocation:\n%s', self.iterations, new_ops, '\n'.join(traceback.format_stack()))
        return True

    def before_iteration(self):
        """Called before each iteration in a Python loop."""
        if self.check_inefficient_unroll and self.iterations > INEFFICIENT_UNROLL_MIN_ITERATIONS:
            self.ops_before_iteration = self._get_ops()
            self.check_op_count_after_iteration = True

    def after_iteration(self):
        """Called after each iteration in a Python loop."""
        self.iterations += 1
        self._check_unroll_limits()
        if self.check_op_count_after_iteration:
            did_warn = self._verify_inefficient_unroll()
            if did_warn:
                self._stop_checking_inefficient_unroll()
            elif self.iterations > INEFFICIENT_UNROLL_MIN_ITERATIONS + 3:
                self._stop_checking_inefficient_unroll()
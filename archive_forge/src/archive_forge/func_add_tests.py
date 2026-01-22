import abc
import itertools
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_model
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import save as save_model
from tensorflow.python.util import nest
def add_tests(test_cls):
    """Add tests for LinearOperator methods."""
    test_name_dict = {'add_to_tensor': _test_add_to_tensor, 'adjoint': _test_adjoint, 'cholesky': _test_cholesky, 'cond': _test_cond, 'composite_tensor': _test_composite_tensor, 'composite_tensor_gradient': _test_composite_tensor_gradient, 'det': _test_det, 'diag_part': _test_diag_part, 'eigvalsh': _test_eigvalsh, 'inverse': _test_inverse, 'log_abs_det': _test_log_abs_det, 'operator_matmul_with_same_type': _test_operator_matmul_with_same_type, 'operator_solve_with_same_type': _test_operator_solve_with_same_type, 'matmul': _test_matmul, 'matmul_with_broadcast': _test_matmul_with_broadcast, 'saved_model': _test_saved_model, 'slicing': _test_slicing, 'solve': _test_solve, 'solve_with_broadcast': _test_solve_with_broadcast, 'to_dense': _test_to_dense, 'trace': _test_trace}
    optional_tests = ['operator_matmul_with_same_type', 'operator_solve_with_same_type']
    tests_with_adjoint_args = ['matmul', 'matmul_with_broadcast', 'solve', 'solve_with_broadcast']
    if set(test_cls.skip_these_tests()).intersection(test_cls.optional_tests()):
        raise ValueError(f"Test class {{test_cls}} had intersecting 'skip_these_tests' {test_cls.skip_these_tests()} and 'optional_tests' {test_cls.optional_tests()}.")
    for name, test_template_fn in test_name_dict.items():
        if name in test_cls.skip_these_tests():
            continue
        if name in optional_tests and name not in test_cls.optional_tests():
            continue
        for dtype, use_placeholder, shape_info in itertools.product(test_cls.dtypes_to_test(), test_cls.use_placeholder_options(), test_cls.operator_shapes_infos()):
            base_test_name = '_'.join(['test', name, '_shape={},dtype={},use_placeholder={}'.format(shape_info.shape, dtype, use_placeholder)])
            if name in tests_with_adjoint_args:
                for adjoint in test_cls.adjoint_options():
                    for adjoint_arg in test_cls.adjoint_arg_options():
                        test_name = base_test_name + ',adjoint={},adjoint_arg={}'.format(adjoint, adjoint_arg)
                        if hasattr(test_cls, test_name):
                            raise RuntimeError('Test %s defined more than once' % test_name)
                        setattr(test_cls, test_name, test_util.run_deprecated_v1(test_template_fn(use_placeholder, shape_info, dtype, adjoint, adjoint_arg, test_cls.use_blockwise_arg())))
            else:
                if hasattr(test_cls, base_test_name):
                    raise RuntimeError('Test %s defined more than once' % base_test_name)
                setattr(test_cls, base_test_name, test_util.run_deprecated_v1(test_template_fn(use_placeholder, shape_info, dtype)))
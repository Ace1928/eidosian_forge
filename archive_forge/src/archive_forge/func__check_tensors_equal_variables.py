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
def _check_tensors_equal_variables(self, obj, tensor_obj):
    """Checks that Variables in `obj` have equivalent Tensors in `tensor_obj."""
    if isinstance(obj, variables.Variable):
        self.assertAllClose(ops.convert_to_tensor(obj), ops.convert_to_tensor(tensor_obj))
    elif isinstance(obj, composite_tensor.CompositeTensor):
        params = getattr(obj, 'parameters', {})
        tensor_params = getattr(tensor_obj, 'parameters', {})
        self.assertAllEqual(params.keys(), tensor_params.keys())
        self._check_tensors_equal_variables(params, tensor_params)
    elif nest.is_mapping(obj):
        for k, v in obj.items():
            self._check_tensors_equal_variables(v, tensor_obj[k])
    elif nest.is_nested(obj):
        for x, y in zip(obj, tensor_obj):
            self._check_tensors_equal_variables(x, y)
    else:
        pass
import functools
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.parallel_for.pfor import PFor
from tensorflow.python.ops.parallel_for.pfor import PForConfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _loop_fn_has_config(loop_fn):
    """Test if `loop_fn` has a `pfor_config` argument."""
    if tf_inspect.isfunction(loop_fn):
        argspec = tf_inspect.getargspec(loop_fn)
        return PFOR_CONFIG_ARG in argspec.args
    elif isinstance(loop_fn, functools.partial):
        fn = loop_fn.func
        argspec = tf_inspect.getargspec(fn)
        return PFOR_CONFIG_ARG in argspec.args and PFOR_CONFIG_ARG not in loop_fn.keywords
    else:
        loop_class = tf_decorator.unwrap(loop_fn)[1]
        if not hasattr(loop_class, '__call__'):
            raise ValueError('`loop_fn` object did not have a __call__ method')
        argspec = tf_inspect.getargspec(loop_class.__call__)
        return PFOR_CONFIG_ARG in argspec.args
import functools
import operator
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import imperative_grad
from tensorflow.python.eager import tape
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _get_arg_spec(f, params, param_args):
    """The positions of the parameters of f to be differentiated in param_args."""
    try:
        args = tf_inspect.getfullargspec(f).args
    except TypeError as e:
        if params is None:
            return range(len(param_args))
        elif all((isinstance(x, int) for x in params)):
            return params
        raise ValueError('Either callable provided is not a function or could not inspect its arguments by name: %s. Original error: %s' % (f, e))
    if params is None:
        if not args:
            return range(len(param_args))
        if args[0] == 'self':
            return range(len(args) - 1)
        else:
            return range(len(args))
    elif all((isinstance(x, str) for x in params)):
        return [args.index(n) for n in params]
    elif all((isinstance(x, int) for x in params)):
        return params
    else:
        raise ValueError('params must be all strings or all integers; got %s.' % params)
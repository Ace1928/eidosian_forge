import inspect
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_decorator
def weak_tensor_unary_op_wrapper(op, x_arg_name=None):
    """Infers input type and adds WeakTensor support to unary ops.

  This wrapper infers input type according to the auto dtype conversion
  semantics - Tensor and NumPy inputs as Tensor of corresponding dtype and
  WeakTensor and python inputs as WeakTensor of corresponding dtype. If the
  inferred input dtype is "weak" and the op doesn't specify a return dtype,
  returns WeakTensor.
  """
    signature = inspect.signature(op)
    if x_arg_name is None:
        arg_names = iter(signature.parameters.keys())
        x_arg_name = next(arg_names)

    def wrapper(*args, **kwargs):
        if not ops.is_auto_dtype_conversion_enabled():
            return op(*args, **kwargs)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        bound_kwargs = bound_arguments.arguments
        x = bound_kwargs[x_arg_name]
        if isinstance(x, tensor.Tensor) or bound_kwargs.get('dtype', None) is not None:
            return op(**bound_kwargs)
        try:
            target_type, is_weak = flexible_dtypes.result_type(x)
        except NotImplementedError:
            logging.warning(f'The new dtype semantics do not support {op.__module__}.{op.__name__}({type(x)}). Falling back to old semantics.')
            return op(**bound_kwargs)
        bound_kwargs[x_arg_name] = _convert_or_cast(x, target_type, 'x')
        return weak_tensor.convert_to_weak_tensor_or_tensor(op(**bound_kwargs), is_weak)
    wrapper = tf_decorator.make_decorator(op, wrapper)
    _update_weak_tensor_patched_ops_in_dispatch_dict(wrapper)
    _TF_UNARY_APIS.append(wrapper)
    return wrapper
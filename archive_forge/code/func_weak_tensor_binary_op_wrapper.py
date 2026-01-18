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
def weak_tensor_binary_op_wrapper(op, y_arg_name=None, special_handling=None):
    """Determines result promotion type and adds WeakTensor support to binary ops.

  This wrapper first infers dtype of any Tensor, WeakTensor, python/numpy
  inputs. Then, both inputs are promoted to the correct promotion result dtype.
  If the result promotion dtype is "weak", returns WeakTensor.
  """
    signature = inspect.signature(op)
    arg_names = iter(signature.parameters.keys())
    x_arg_name = next(arg_names)
    if y_arg_name is None:
        y_arg_name = next(arg_names)

    def wrapper(*args, **kwargs):
        if not ops.is_auto_dtype_conversion_enabled():
            return op(*args, **kwargs)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        bound_kwargs = bound_arguments.arguments
        x = bound_kwargs[x_arg_name]
        y = bound_kwargs[y_arg_name]
        try:
            target_type, is_weak = flexible_dtypes.result_type(x, y)
        except NotImplementedError:
            logging.warning(f'The new dtype semantics do not support {op.__module__}.{op.__name__}({type(x)}, {type(y)}). Falling back to old semantics.')
            return op(**bound_kwargs)
        if special_handling == 'variable_method':
            if target_type != x.dtype:
                raise TypeError(f'Variable dtype is immutable. Calling {op.__name__} of Variable (with dtype {x.dtype}) on {y} requires converting {y} to {x.dtype}. This is disabled in the current promotion semantics. Please convert {y} manually before calling {op.__name__}.')
            bound_kwargs[y_arg_name] = _convert_or_cast(y, target_type, 'y')
            return op(**bound_kwargs)
        elif special_handling == 'constant':
            if isinstance(x, weak_tensor.WeakTensor):
                bound_kwargs[x_arg_name] = x.to_tensor()
            if y is not None:
                is_weak = False
                if target_type != y:
                    return op(**bound_kwargs)
                if isinstance(x, core.Tensor):
                    bound_kwargs[x_arg_name] = _convert_or_cast(x, target_type, 'x')
            else:
                bound_kwargs['dtype'] = target_type
        else:
            bound_kwargs[x_arg_name] = _convert_or_cast(x, target_type, 'x')
            bound_kwargs[y_arg_name] = _convert_or_cast(y, target_type, 'y')
        return weak_tensor.convert_to_weak_tensor_or_tensor(op(**bound_kwargs), is_weak)
    wrapper = tf_decorator.make_decorator(op, wrapper)
    _update_weak_tensor_patched_ops_in_dispatch_dict(wrapper)
    _TF_BINARY_APIS.append(wrapper)
    return wrapper
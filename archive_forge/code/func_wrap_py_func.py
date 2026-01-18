from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_ops
def wrap_py_func(f, args, kwargs=None):
    """Helper that wraps a callable to py_func.

  The helper passes tensor arguments through the py_func interface. Non-tensor
  arguments are allowed, and will be passed to f directly. Note that non-tensor
  arguments are captured by f will not update every time the wrapper is
  called (this is consistent with its argument list, which only includes
  the tensor arguments). In general, it's safest not to reuse this wrapper.

  Args:
    f: Callable
    args: Positional arguments for f, as list or tuple.
    kwargs: Keyword arguments for f, as dict with string keys. May be None.

  Returns:
    The return values of f converted to tensor.
  Raises:
    ValueError: if any of the arguments are incorrect.
  """
    tensor_args = []
    tensor_args_idx = {}
    n_args = len(args)
    arg_is_tensor = tuple(map(tensor_util.is_tf_type, args))
    for i in range(n_args):
        if arg_is_tensor[i]:
            tensor_args_idx[i] = len(tensor_args)
            tensor_args.append(args[i])
    if kwargs:
        kwarg_keys = tuple(kwargs.keys())
        kwarg_is_tensor = {k: tensor_util.is_tf_type(kwargs[k]) for k in kwarg_keys}
        for k in kwarg_keys:
            if kwarg_is_tensor[k]:
                tensor_args_idx[k] = len(tensor_args)
                tensor_args.append(kwargs[k])
    else:
        kwarg_keys = ()

    def f_wrapper(*tensor_args):
        f_args = tuple((tensor_args[tensor_args_idx[i]] if arg_is_tensor[i] else a for i, a in enumerate(args)))
        f_kwargs = {k: tensor_args[tensor_args_idx[k]] if kwarg_is_tensor[k] else kwargs[k] for i, k in enumerate(kwarg_keys)}
        f(*f_args, **f_kwargs)
        return 1
    return script_ops.eager_py_func(f_wrapper, tensor_args, dtypes.int32)
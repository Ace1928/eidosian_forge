from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_ops
def _tf_py_func_print(*objects, **kwargs):
    """Overload of print_ as a py_func implementation."""
    override_kwargs = {k: v for k, v in kwargs.items() if v is not py_builtins.UNSPECIFIED}
    if 'flush' not in override_kwargs:
        override_kwargs['flush'] = True

    def print_wrapper(*vals, **kwargs):
        vals = tuple((v.numpy() if tensor_util.is_tf_type(v) else v for v in vals))
        vals = tuple((v.decode('utf-8') if isinstance(v, bytes) else v for v in vals))
        print(*vals, **kwargs)
    return wrap_py_func(print_wrapper, objects, override_kwargs)
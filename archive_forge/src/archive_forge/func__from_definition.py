import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
def _from_definition(fdef, grad_func=None):
    """Creates a _DefinedFunction initialized from a FunctionDef proto.

  Args:
    fdef: a FunctionDef
    grad_func: a _DefinedFunction or None

  Returns:
    A _DefinedFunction representing fdef
  """
    func = None
    argnames = [arg.name for arg in fdef.signature.input_arg]
    input_types = tuple((dtypes.as_dtype(arg.type) for arg in fdef.signature.input_arg))
    func_name = fdef.signature.name
    python_grad_func = None
    out_names = [arg.name for arg in fdef.signature.output_arg]
    result = _DefinedFunction(func, argnames, input_types, func_name, grad_func, python_grad_func, out_names)
    serialized = fdef.SerializeToString()
    c_func = c_api.TF_FunctionImportFunctionDef(serialized)
    result._c_func = c_api_util.ScopedTFFunction(c_func, func_name)
    result._extra_inputs = []
    result._op_def = fdef.signature
    return result
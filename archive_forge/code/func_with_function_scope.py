from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.operators import variables
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest
def with_function_scope(thunk, scope_name, options):
    """Inline version of the FunctionScope context manager."""
    with FunctionScope('lambda_', scope_name, options) as scope:
        return thunk(scope)
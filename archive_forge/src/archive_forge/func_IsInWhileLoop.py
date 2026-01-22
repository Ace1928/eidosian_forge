from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import variables
def IsInWhileLoop(op):
    ctxt = op._get_control_flow_context()
    return GetContainingWhileContext(ctxt) is not None
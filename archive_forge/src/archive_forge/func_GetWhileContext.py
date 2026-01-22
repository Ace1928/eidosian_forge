import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def GetWhileContext(op):
    """Get the WhileContext to which this op belongs."""
    ctxt = op._get_control_flow_context()
    if ctxt:
        ctxt = ctxt.GetWhileContext()
    return ctxt
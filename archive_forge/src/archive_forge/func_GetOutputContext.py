import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def GetOutputContext(op):
    """Return the control flow context for the output of an op."""
    ctxt = op._get_control_flow_context()
    if ctxt is not None and IsLoopExit(op):
        ctxt = ctxt.outer_context
    return ctxt
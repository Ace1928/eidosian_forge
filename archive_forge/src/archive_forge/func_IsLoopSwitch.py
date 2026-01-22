import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsLoopSwitch(op):
    """Return true if `op` is the Switch for a while loop."""
    if IsSwitch(op):
        ctxt = op._get_control_flow_context()
        return ctxt is not None and ctxt.IsWhileContext() and (not IsCondSwitch(op))
    return False
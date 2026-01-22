import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def GetContainingCondContext(ctxt):
    """Returns the first ancestor CondContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a CondContext, or None if `ctxt` is not in a cond.

  Args:
    ctxt: ControlFlowContext

  Returns:
    `ctxt` if `ctxt` is a CondContext, the most nested CondContext containing
    `ctxt`, or None if `ctxt` is not in a cond.
  """
    while ctxt:
        if ctxt.IsCondContext():
            return ctxt
        ctxt = ctxt.outer_context
    return None
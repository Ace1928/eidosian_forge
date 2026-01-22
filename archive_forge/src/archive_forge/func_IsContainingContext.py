import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsContainingContext(ctxt, maybe_containing_ctxt):
    """Returns true if `maybe_containing_ctxt` is or contains `ctxt`."""
    while ctxt is not maybe_containing_ctxt:
        if ctxt is None:
            return False
        ctxt = ctxt.outer_context
    return True
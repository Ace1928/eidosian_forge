import enum
import inspect
import threading
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.util.tf_export import tf_export
def _default_control_status_ctx():
    return ControlStatusCtx(status=Status.UNSPECIFIED)
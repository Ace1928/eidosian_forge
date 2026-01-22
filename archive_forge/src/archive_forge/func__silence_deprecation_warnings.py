import contextlib
from tensorboard import data_compat
from tensorboard import dataclass_compat
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.util import platform_util
from tensorboard.util import tb_logging
def _silence_deprecation_warnings():
    """Context manager that best-effort silences TF deprecation warnings."""
    try:
        from tensorflow.python.util import deprecation
        return deprecation.silence()
    except (ImportError, AttributeError):
        return _NULLCONTEXT
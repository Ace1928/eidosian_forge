import contextlib
from tensorboard import data_compat
from tensorboard import dataclass_compat
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.util import platform_util
from tensorboard.util import tb_logging
@contextlib.contextmanager
def _nullcontext():
    """Pre-Python-3.7-compatible standin for contextlib.nullcontext."""
    yield
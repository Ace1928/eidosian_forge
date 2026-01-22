import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _convert_scalar_event(event):
    """Helper for `read_scalars`."""
    return provider.ScalarDatum(step=event.step, wall_time=event.wall_time, value=tensor_util.make_ndarray(event.tensor_proto).item())
import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _validate_downsample(self, downsample):
    if downsample is None:
        raise TypeError('`downsample` required but not given')
    if isinstance(downsample, int):
        return
    raise TypeError('`downsample` must be an int, but got %r: %r' % (type(downsample), downsample))
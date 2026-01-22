import numpy as np
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.tensorflow_stub import dtypes, compat, tensor_shape
def _NotNone(v):
    if v is None:
        return _Message('None')
    else:
        return v
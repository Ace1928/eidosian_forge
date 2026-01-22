import numpy as np
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.tensorflow_stub import dtypes, compat, tensor_shape
def _FirstNotNone(l):
    for x in l:
        if x is not None:
            return x
    return None
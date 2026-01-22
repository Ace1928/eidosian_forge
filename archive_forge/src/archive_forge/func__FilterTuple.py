import numpy as np
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.tensorflow_stub import dtypes, compat, tensor_shape
def _FilterTuple(v):
    if not isinstance(v, (list, tuple)):
        return v
    if isinstance(v, tuple):
        if not any((isinstance(x, (list, tuple)) for x in v)):
            return None
    if isinstance(v, list):
        if not any((isinstance(x, (list, tuple)) for x in v)):
            return _FirstNotNone([None if isinstance(x, (list, tuple)) else x for x in v])
    return _FirstNotNone([_FilterTuple(x) for x in v])
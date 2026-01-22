import numpy as np
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.tensorflow_stub import dtypes, compat, tensor_shape
def _FilterInt(v):
    if isinstance(v, (list, tuple)):
        return _FirstNotNone([_FilterInt(x) for x in v])
    return None if isinstance(v, (compat.integral_types, tensor_shape.Dimension)) else _NotNone(v)
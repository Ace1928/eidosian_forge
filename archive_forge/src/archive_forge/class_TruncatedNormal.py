from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
class TruncatedNormal(init_ops.TruncatedNormal):

    def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
        super(TruncatedNormal, self).__init__(mean=mean, stddev=stddev, seed=seed, dtype=dtype)
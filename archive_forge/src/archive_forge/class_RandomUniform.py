from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
class RandomUniform(init_ops.RandomUniform):

    def __init__(self, minval=-0.05, maxval=0.05, seed=None, dtype=dtypes.float32):
        super(RandomUniform, self).__init__(minval=minval, maxval=maxval, seed=seed, dtype=dtype)
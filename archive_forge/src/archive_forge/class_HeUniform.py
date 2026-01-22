from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
class HeUniform(init_ops.VarianceScaling):

    def __init__(self, seed=None):
        super(HeUniform, self).__init__(scale=2.0, mode='fan_in', distribution='uniform', seed=seed)

    def get_config(self):
        return {'seed': self.seed}
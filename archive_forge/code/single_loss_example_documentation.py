from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import step_fn
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
A model that uses batchnorm.
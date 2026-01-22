import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _rebatch(input_dataset, batch_size, drop_remainder=False, name=None):
    return _RebatchDataset(input_dataset, batch_size, drop_remainder, name)
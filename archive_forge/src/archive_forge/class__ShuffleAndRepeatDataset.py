import functools
import numpy as np
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class _ShuffleAndRepeatDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that fuses `shuffle` and `repeat`."""

    def __init__(self, input_dataset, buffer_size, count=None, seed=None):
        self._input_dataset = input_dataset
        self._buffer_size = ops.convert_to_tensor(buffer_size, dtype=dtypes.int64, name='buffer_size')
        if count is None:
            self._count = constant_op.constant(-1, dtype=dtypes.int64, name='count')
        else:
            self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name='count')
        self._seed, self._seed2 = random_seed.get_seed(seed)
        variant_tensor = gen_dataset_ops.shuffle_and_repeat_dataset(self._input_dataset._variant_tensor, buffer_size=self._buffer_size, count=self._count, seed=self._seed, seed2=self._seed2, **self._flat_structure)
        super(_ShuffleAndRepeatDataset, self).__init__(input_dataset, variant_tensor)
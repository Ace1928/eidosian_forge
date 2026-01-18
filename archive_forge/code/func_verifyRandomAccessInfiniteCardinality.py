import os
import random
import re
from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import test_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
def verifyRandomAccessInfiniteCardinality(self, dataset, expected):
    """Tests randomly accessing elements of a dataset."""
    len_expected = len(expected)
    indices = list(range(len_expected)) * 2
    random.shuffle(indices)
    for i in indices:
        self.assertAllEqual(expected[i], self.evaluate(random_access.at(dataset, i)))
    indices = set(sorted(indices))
    for i in indices:
        self.assertAllEqual(expected[i], self.evaluate(random_access.at(dataset, i)))
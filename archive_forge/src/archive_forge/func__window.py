from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
def _window(input_dataset, size, shift, stride, drop_remainder, name):
    if shift is None:
        shift = size
    return _WindowDataset(input_dataset, size, shift, stride, drop_remainder, name=name)
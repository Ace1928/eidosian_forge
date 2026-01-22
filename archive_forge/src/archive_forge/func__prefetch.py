from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
def _prefetch(input_dataset, buffer_size, name=None):
    """See `Dataset.prefetch()` for details."""
    if debug_mode.DEBUG_MODE:
        return input_dataset
    return _PrefetchDataset(input_dataset, buffer_size, name=name)
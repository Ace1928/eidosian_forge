import warnings
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
def _map_v1_with_legacy_function(input_dataset, map_func, num_parallel_calls=None, deterministic=None):
    """See `Dataset.map()` for details."""
    if num_parallel_calls is None:
        if deterministic is not None:
            warnings.warn('The `deterministic` argument has no effect unless the `num_parallel_calls` argument is specified.')
        return dataset_ops.DatasetV1Adapter(_MapDataset(input_dataset, map_func, preserve_cardinality=False, use_legacy_function=True))
    else:
        return dataset_ops.DatasetV1Adapter(_ParallelMapDataset(input_dataset, map_func, num_parallel_calls, deterministic, preserve_cardinality=False, use_legacy_function=True))
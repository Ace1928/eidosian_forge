from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _group_by_window(input_dataset, key_func, reduce_func, window_size=None, window_size_func=None, name=None):
    """See `Dataset.group_by_window()` for details."""
    if window_size is not None and window_size_func or not (window_size is not None or window_size_func):
        raise ValueError('Either the `window_size` argument or the `window_size_func` argument must be specified.')
    if window_size is not None:

        def constant_window_func(unused_key):
            return ops.convert_to_tensor(window_size, dtype=dtypes.int64)
        window_size_func = constant_window_func
    assert window_size_func is not None
    return _GroupByWindowDataset(input_dataset, key_func, reduce_func, window_size_func, name=name)
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops
def _range(*args, **kwargs):
    return _RangeDataset(*args, **kwargs)
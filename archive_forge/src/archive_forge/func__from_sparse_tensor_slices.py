from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops
def _from_sparse_tensor_slices(sparse_tensor):
    return dataset_ops.DatasetV1Adapter(_SparseTensorSliceDataset(sparse_tensor))
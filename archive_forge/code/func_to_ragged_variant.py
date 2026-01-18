from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops.ragged import ragged_tensor
def to_ragged_variant(value):
    """Re-encode Tensors as RaggedTensors."""
    if not isinstance(value, tensor.Tensor) or value.shape.rank is None or value.shape.is_fully_defined():
        return value
    else:
        spec = to_ragged_spec(tensor.TensorSpec.from_tensor(value))
        if spec._ragged_rank > 0:
            value = ragged_tensor.RaggedTensor.from_tensor(value, ragged_rank=spec._ragged_rank)
        return spec._to_tensor_list(value)[0]
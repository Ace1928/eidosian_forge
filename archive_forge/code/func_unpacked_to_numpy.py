from typing import List
import numpy as np
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.types.core import Tensor, TensorLike  # pylint: disable=g-multiple-import
def unpacked_to_numpy(unpacked: List[TensorLike], layout: layout_lib.Layout) -> np.ndarray:
    """Heals local Tensor components to a numpy array."""
    if len(unpacked) != len(layout.offset_to_shard()):
        raise ValueError('Wrong number of component Tensors.')
    unravelled = np.ndarray([layout.num_shards(i) for i in range(layout.rank)], dtype=object)
    for offset, loc in enumerate(layout.offset_to_shard()):
        unravelled[loc] = unpacked[offset]
    concat_tensor = np.block(unravelled.tolist())
    while concat_tensor.ndim > unpacked[0].ndim:
        concat_tensor = np.squeeze(concat_tensor, axis=0)
    return concat_tensor
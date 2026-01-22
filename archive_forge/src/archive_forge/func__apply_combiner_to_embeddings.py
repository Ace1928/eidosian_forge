from typing import Any, Dict, Iterable, Optional, Text, Union
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _apply_combiner_to_embeddings(self, embeddings: tensor.Tensor, weight: tensor.Tensor, combiner: Optional[Text]=None) -> tensor.Tensor:
    """Apply the combiner to the embedding look up result on second to last axis.

    Args:
      embeddings: A Tensor of the embedding lookup result.
      weight: A Tensor of weight which has the same shape of the embeddings.
      combiner: One of "mean", "sum", "sqrtn". Defaults to "mean".

    Raises:
      ValueError: If the combiner is not one of 'mean', 'sqrtn' or 'sum'.
    Returns:
      A Tensor.
    """
    if combiner is None:
        combiner = 'mean'
    if combiner == 'sum':
        embeddings = math_ops.reduce_sum(embeddings, axis=-2)
    elif combiner == 'mean':
        embeddings = math_ops.reduce_sum(embeddings, axis=-2)
        weight_sum = math_ops.reduce_sum(weight, axis=-2)
        embeddings = math_ops.div_no_nan(embeddings, weight_sum)
    elif combiner == 'sqrtn':
        embeddings = math_ops.reduce_sum(embeddings, axis=-2)
        weight_squared = math_ops.pow(weight, 2)
        weight_sum = math_ops.reduce_sum(weight_squared, axis=-2)
        weight_sum_sqrt = math_ops.sqrt(weight_sum)
        embeddings = math_ops.div_no_nan(embeddings, weight_sum_sqrt)
    else:
        raise ValueError(f"combiner must be one of 'mean', 'sqrtn' or 'sum', got {combiner}")
    return embeddings
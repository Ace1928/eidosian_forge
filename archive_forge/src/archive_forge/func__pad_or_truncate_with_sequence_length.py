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
def _pad_or_truncate_with_sequence_length(self, embeddings: tensor.Tensor, sequence_length: int) -> tensor.Tensor:
    """Pad or truncate the embedding lookup result based on the sequence length.

    Args:
      embeddings: A rank 3 Tensor of the embedding lookup result.
      sequence_length: number of the max sequence length set in the feature
        config.

    Returns:
      A Tensor with second last axis padded or truncated.
    """
    original_sequence_length = embeddings.shape[1]
    if original_sequence_length > sequence_length:
        embeddings = array_ops.slice(embeddings, begin=[0, 0, 0], size=[-1, sequence_length, -1])
    else:
        embeddings = array_ops.pad(embeddings, paddings=[[0, 0], [0, sequence_length - original_sequence_length], [0, 0]])
    return embeddings
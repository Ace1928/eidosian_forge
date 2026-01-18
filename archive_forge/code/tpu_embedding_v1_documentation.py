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
Embedding lookup for ragged tensor based on its feature config.

    Args:
      inp: a single rank 2 RaggedTensor input.
      weight: None or RaggedTensor which has the same shape of the input.
      table: a table variable.
      feature: a feature config.

    Returns:
      Embedding lookup result.

    Raises:
      ValueError: if input ragged tensor is not rank 2 or output shape set in
      the feature config doesn't match with the first dim size of the input.
    
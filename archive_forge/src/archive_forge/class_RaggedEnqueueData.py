import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
class RaggedEnqueueData(collections.namedtuple('RaggedEnqueueData', ['embedding_indices', 'row_splits', 'aggregation_weights'])):
    """RaggedTensor Data to be enqueued through generate_enqueue_ops()."""

    def __new__(cls, embedding_indices, row_splits=None, aggregation_weights=None):
        """Data to be enqueued through generate_enqueue_ops().

    Args:
      embedding_indices: A rank 1 Tensor, indices into the embedding tables. It
        corresponds to ids.values in embedding_lookup(), when ids is a
        RaggedTensor. Both int32 and int64 are allowed and will be converted to
        int32 internally.
      row_splits: A rank 1 Tensor specifying the length of  the break points for
        splitting embedding_indices and aggregation_weights. It corresponds to
        ids.row_splits in embedding_lookup(), when ids is a RaggedTensor. Both
        int32 and int64 are allowed and will be converted to int32 internally.
      aggregation_weights: A rank 1 Tensor containing per training example
        aggregation weights. It corresponds to the values field of a
        RaggedTensor with the same row_splits as ids in embedding_lookup(), when
        ids is a RaggedTensor.

    Returns:
      An RaggedEnqueueData tuple.

    """
        return super().__new__(cls, embedding_indices, row_splits, aggregation_weights)

    @staticmethod
    def from_ragged_tensor(rg_tensor, weights=None):
        return RaggedEnqueueData(rg_tensor.values, rg_tensor.row_splits, aggregation_weights=weights.values if weights is not None else None)
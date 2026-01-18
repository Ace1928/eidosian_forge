from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
A placeholder op for enqueueing embedding IDs to the TPU.

  Args:
    sample_indices_or_row_splits: A list of rank 1 or 2 Tensors. When rank 2,
      the tensors specify the training example to which the corresponding
      embedding_indices and aggregation_weights values belong. If the size of
      its first dimension is 0, we assume each embedding_indices belongs to a
      different sample. Both int32 and int64 are allowed and will be converted
      to int32 internally. When rank 1, the tensors specify the row splits for
      splitting embedding_indices and aggregation_weights into rows. It
      corresponds to ids.row_splits in embedding_lookup(), when ids is a
      RaggedTensor. When enqueuing N-D ragged tensor, only the last dimension is
      allowed to be ragged. the row splits is 1-D dense tensor. When empty, we
      assume a dense tensor is passed to the op. Both int32 and int64 are
      allowed and will be converted to int32 internally.
    embedding_indices: A list of rank 1 Tensors, indices into the embedding
      tables. Both int32 and int64 are allowed and will be converted to int32
      internally.
    aggregation_weights: A list of rank 1 Tensors containing per training
      example aggregation weights. Both float32 and float64 are allowed and will
      be converted to float32 internally.
    device_ordinal: The TPU device to use. Should be >= 0 and less than the
      number of TPU cores in the task on which the node is placed.
    combiners: A list of string scalars, one for each embedding table that
      specify how to normalize the embedding activations after weighted
      summation. Supported combiners are 'mean', 'sum', or 'sqrtn'. It is
      invalid to have the sum of the weights be 0 for 'mean' or the sum of the
      squared weights be 0 for 'sqrtn'. If combiners isn't passed, the default
      is to use 'sum' for all tables (optional).
    mode_override: A string input that overrides the mode specified in the
      TPUEmbeddingConfiguration. Supported values are {'unspecified',
      'inference', 'training', 'backward_pass_only'}. When set to 'unspecified',
      the mode set in TPUEmbeddingConfiguration is used, otherwise mode_override
      is used (optional).
    name: A name for the operation (optional).

  Returns:
    An EnqueueTPUEmbeddingArbitraryTensorBatch operation.
  
import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops
Get gradients wrt the activations of each feature.

  Args:
    tpu_embedding: TPUEmbedding, create dummy table variable to be used with
      tpu_embedding.

  Returns:
    An OrderedDict mapping feature name to gradient.

  Raises:
    ValueError: if some gradients are not defined.
  
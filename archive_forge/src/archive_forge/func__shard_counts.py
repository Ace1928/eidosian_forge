import dataclasses
import operator
from typing import Any, List, Optional, Sequence, Tuple
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _shard_counts(layout: layout_lib.Layout, batch_dim: Optional[str]=None) -> List[int]:
    """Computes a list of the number of shards in each dimension of the layout.

  The shard counts are used to slice each dataset element. The batch dimension's
  count is overridden to 1 since we only consider how many shards to make
  locally (within each local replica). Sharding across clients is handled by
  either tf.data.Dataset's shard transformation (in the single-client case) or
  tf.data service's distribute function (in the multi-client case).

  Args:
    layout: the layout to compute the shard counts for.
    batch_dim: the name of the batch dimension of the layout, if present.

  Returns:
    A list of shard counts, one element per dimension of the layout.
  """
    shard_counts = []
    for spec in layout.sharding_specs:
        if spec in (batch_dim, layout_lib.UNSHARDED):
            shard_counts.append(1)
        else:
            shard_counts.append(layout.mesh.dim_size(spec))
    return shard_counts
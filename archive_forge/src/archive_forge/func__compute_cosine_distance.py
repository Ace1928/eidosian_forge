from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as random_seed_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_clustering_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.ops.gen_clustering_ops import *
@classmethod
def _compute_cosine_distance(cls, inputs, clusters, inputs_normalized=True):
    """Computes cosine distance between each input and each cluster center.

    Args:
      inputs: list of input Tensor.
      clusters: cluster Tensor
      inputs_normalized: if True, it assumes that inp and clusters are
        normalized and computes the dot product which is equivalent to the
        cosine distance. Else it L2 normalizes the inputs first.

    Returns:
      list of Tensors, where each element corresponds to each element in inp.
      The value is the distance of each row to all the cluster centers.
    """
    output = []
    if not inputs_normalized:
        with ops.colocate_with(clusters, ignore_existing=True):
            clusters = nn_impl.l2_normalize(clusters, axis=1)
    for inp in inputs:
        with ops.colocate_with(inp, ignore_existing=True):
            if not inputs_normalized:
                inp = nn_impl.l2_normalize(inp, axis=1)
            output.append(1 - math_ops.matmul(inp, clusters, transpose_b=True))
    return output
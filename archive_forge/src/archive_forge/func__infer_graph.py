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
def _infer_graph(self, inputs, clusters):
    """Maps input to closest cluster and the score.

    Args:
      inputs: list of input Tensors.
      clusters: Tensor of cluster centers.

    Returns:
      List of tuple, where each value in tuple corresponds to a value in inp.
      The tuple has following three elements:
      all_scores: distance of each input to each cluster center.
      score: distance of each input to closest cluster center.
      cluster_idx: index of cluster center closest to the corresponding input.
    """
    assert isinstance(inputs, list)
    scores = self._distance_graph(inputs, clusters, self._distance_metric)
    output = []
    if self._distance_metric == COSINE_DISTANCE and (not self._clusters_l2_normalized()):
        with ops.colocate_with(clusters, ignore_existing=True):
            clusters = nn_impl.l2_normalize(clusters, axis=1)
    for inp, score in zip(inputs, scores):
        with ops.colocate_with(inp, ignore_existing=True):
            indices, distances = gen_clustering_ops.nearest_neighbors(inp, clusters, 1)
            if self._distance_metric == COSINE_DISTANCE:
                distances *= 0.5
            output.append((score, array_ops.squeeze(distances, [-1]), array_ops.squeeze(indices, [-1])))
    return zip(*output)
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
def training_graph(self):
    """Generate a training graph for kmeans algorithm.

    This returns, among other things, an op that chooses initial centers
    (init_op), a boolean variable that is set to True when the initial centers
    are chosen (cluster_centers_initialized), and an op to perform either an
    entire Lloyd iteration or a mini-batch of a Lloyd iteration (training_op).
    The caller should use these components as follows. A single worker should
    execute init_op multiple times until cluster_centers_initialized becomes
    True. Then multiple workers may execute training_op any number of times.

    Returns:
      A tuple consisting of:
      all_scores: A matrix (or list of matrices) of dimensions (num_input,
        num_clusters) where the value is the distance of an input vector and a
        cluster center.
      cluster_idx: A vector (or list of vectors). Each element in the vector
        corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      scores: Similar to cluster_idx but specifies the distance to the
        assigned cluster instead.
      cluster_centers_initialized: scalar indicating whether clusters have been
        initialized.
      init_op: an op to initialize the clusters.
      training_op: an op that runs an iteration of training.
    """
    if isinstance(self._initial_clusters, str) or callable(self._initial_clusters):
        initial_clusters = self._initial_clusters
        num_clusters = ops.convert_to_tensor(self._num_clusters)
    else:
        initial_clusters = ops.convert_to_tensor(self._initial_clusters)
        num_clusters = array_ops.shape(initial_clusters)[0]
    inputs = self._inputs
    cluster_centers_var, cluster_centers_initialized, total_counts, cluster_centers_updated, update_in_steps = self._create_variables(num_clusters)
    init_op = _InitializeClustersOpFactory(self._inputs, num_clusters, initial_clusters, self._distance_metric, self._seed, self._kmeans_plus_plus_num_retries, self._kmc2_chain_length, cluster_centers_var, cluster_centers_updated, cluster_centers_initialized).op()
    cluster_centers = cluster_centers_var
    if self._distance_metric == COSINE_DISTANCE:
        inputs = self._l2_normalize_data(inputs)
        if not self._clusters_l2_normalized():
            cluster_centers = nn_impl.l2_normalize(cluster_centers, dim=1)
    all_scores, scores, cluster_idx = self._infer_graph(inputs, cluster_centers)
    if self._use_mini_batch:
        sync_updates_op = self._mini_batch_sync_updates_op(update_in_steps, cluster_centers_var, cluster_centers_updated, total_counts)
        assert sync_updates_op is not None
        with ops.control_dependencies([sync_updates_op]):
            training_op = self._mini_batch_training_op(inputs, cluster_idx, cluster_centers_updated, total_counts)
    else:
        assert cluster_centers == cluster_centers_var
        training_op = self._full_batch_training_op(inputs, num_clusters, cluster_idx, cluster_centers_var)
    return (all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, training_op)
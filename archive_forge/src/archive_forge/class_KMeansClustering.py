from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clustering_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
@estimator_export(v1=['estimator.experimental.KMeans'])
class KMeansClustering(estimator.Estimator):
    """An Estimator for K-Means clustering.

  Example:
  ```
  import numpy as np
  import tensorflow as tf

  num_points = 100
  dimensions = 2
  points = np.random.uniform(0, 1000, [num_points, dimensions])

  def input_fn():
    return tf.compat.v1.train.limit_epochs(
        tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

  num_clusters = 5
  kmeans = tf.compat.v1.estimator.experimental.KMeans(
      num_clusters=num_clusters, use_mini_batch=False)

  # train
  num_iterations = 10
  previous_centers = None
  for _ in xrange(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
      print 'delta:', cluster_centers - previous_centers
    previous_centers = cluster_centers
    print 'score:', kmeans.score(input_fn)
  print 'cluster centers:', cluster_centers

  # map the input points to their clusters
  cluster_indices = list(kmeans.predict_cluster_index(input_fn))
  for i, point in enumerate(points):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    print 'point:', point, 'is in cluster', cluster_index, 'centered at', center
  ```

  The `SavedModel` saved by the `export_saved_model` method does not include the
  cluster centers. However, the cluster centers may be retrieved by the
  latest checkpoint saved during training. Specifically,
  ```
  kmeans.cluster_centers()
  ```
  is equivalent to
  ```
  tf.train.load_variable(
      kmeans.model_dir, KMeansClustering.CLUSTER_CENTERS_VAR_NAME)
  ```
  """
    SQUARED_EUCLIDEAN_DISTANCE = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE
    COSINE_DISTANCE = clustering_ops.COSINE_DISTANCE
    RANDOM_INIT = clustering_ops.RANDOM_INIT
    KMEANS_PLUS_PLUS_INIT = clustering_ops.KMEANS_PLUS_PLUS_INIT
    SCORE = 'score'
    CLUSTER_INDEX = 'cluster_index'
    ALL_DISTANCES = 'all_distances'
    CLUSTER_CENTERS_VAR_NAME = clustering_ops.CLUSTERS_VAR_NAME

    def __init__(self, num_clusters, model_dir=None, initial_clusters=RANDOM_INIT, distance_metric=SQUARED_EUCLIDEAN_DISTANCE, seed=None, use_mini_batch=True, mini_batch_steps_per_iteration=1, kmeans_plus_plus_num_retries=2, relative_tolerance=None, config=None, feature_columns=None):
        """Creates an Estimator for running KMeans training and inference.

    This Estimator implements the following variants of the K-means algorithm:

    If `use_mini_batch` is False, it runs standard full batch K-means. Each
    training step runs a single iteration of K-Means and must process the full
    input at once. To run in this mode, the `input_fn` passed to `train` must
    return the entire input dataset.

    If `use_mini_batch` is True, it runs a generalization of the mini-batch
    K-means algorithm. It runs multiple iterations, where each iteration is
    composed of `mini_batch_steps_per_iteration` steps. Each training step
    accumulates the contribution from one mini-batch into temporary storage.
    Every `mini_batch_steps_per_iteration` steps, the cluster centers are
    updated and the temporary storage cleared for the next iteration.
    For example: the entire dataset contains 64k examples, where the batch size
    is 64. User can choose mini_batch_steps_per_iteration = 100 to run 10% of
    the entire data every iteration in order to update the cluster centers.
    Note that:
      * If `mini_batch_steps_per_iteration=1`, the algorithm reduces to the
        standard K-means mini-batch algorithm.
      * If `mini_batch_steps_per_iteration = num_inputs / batch_size`, the
        algorithm becomes an asynchronous version of the full-batch algorithm.
        However, there is no guarantee by this implementation that each input
        is seen exactly once per iteration. Also, different updates are applied
        asynchronously without locking. So this asynchronous version may not
        behave exactly like a full-batch version.

    Args:
      num_clusters: An integer tensor specifying the number of clusters. This
        argument is ignored if `initial_clusters` is a tensor or numpy array.
      model_dir: The directory to save the model results and log files.
      initial_clusters: Specifies how the initial cluster centers are chosen.
        One of the following: * a tensor or numpy array with the initial cluster
          centers. * a callable `f(inputs, k)` that selects and returns up to
          `k` centers from an input batch. `f` is free to return any number of
          centers from `0` to `k`. It will be invoked on successive input
          batches as necessary until all `num_clusters` centers are chosen.
        * `KMeansClustering.RANDOM_INIT`: Choose centers randomly from an input
          batch. If the batch size is less than `num_clusters` then the entire
          batch is chosen to be initial cluster centers and the remaining
          centers are chosen from successive input batches.
        * `KMeansClustering.KMEANS_PLUS_PLUS_INIT`: Use kmeans++ to choose
          centers from the first input batch. If the batch size is less than
          `num_clusters`, a TensorFlow runtime error occurs.
      distance_metric: The distance metric used for clustering. One of:
        * `KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE`: Euclidean distance
          between vectors `u` and `v` is defined as \\\\(||u - v||_2\\\\) which is
          the square root of the sum of the absolute squares of the elements'
          difference.
        * `KMeansClustering.COSINE_DISTANCE`: Cosine distance between vectors
          `u` and `v` is defined as \\\\(1 - (u . v) / (||u||_2 ||v||_2)\\\\).
      seed: Python integer. Seed for PRNG used to initialize centers.
      use_mini_batch: A boolean specifying whether to use the mini-batch k-means
        algorithm. See explanation above.
      mini_batch_steps_per_iteration: The number of steps after which the
        updated cluster centers are synced back to a master copy. Used only if
        `use_mini_batch=True`. See explanation above.
      kmeans_plus_plus_num_retries: For each point that is sampled during
        kmeans++ initialization, this parameter specifies the number of
        additional points to draw from the current distribution before selecting
        the best. If a negative value is specified, a heuristic is used to
        sample `O(log(num_to_sample))` additional points. Used only if
        `initial_clusters=KMeansClustering.KMEANS_PLUS_PLUS_INIT`.
      relative_tolerance: A relative tolerance of change in the loss between
        iterations. Stops learning if the loss changes less than this amount.
        This may not work correctly if `use_mini_batch=True`.
      config: See `tf.estimator.Estimator`.
      feature_columns: An optionable iterable containing all the feature columns
        used by the model. All items in the set should be feature column
        instances that can be passed to `tf.feature_column.input_layer`. If this
        is None, all features will be used.

    Raises:
      ValueError: An invalid argument was passed to `initial_clusters` or
        `distance_metric`.
    """
        if isinstance(initial_clusters, str) and initial_clusters not in [KMeansClustering.RANDOM_INIT, KMeansClustering.KMEANS_PLUS_PLUS_INIT]:
            raise ValueError("Unsupported initialization algorithm '%s'" % initial_clusters)
        if distance_metric not in [KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE, KMeansClustering.COSINE_DISTANCE]:
            raise ValueError("Unsupported distance metric '%s'" % distance_metric)
        self._distance_metric = distance_metric
        super(KMeansClustering, self).__init__(model_fn=_ModelFn(num_clusters, initial_clusters, distance_metric, seed, use_mini_batch, mini_batch_steps_per_iteration, kmeans_plus_plus_num_retries, relative_tolerance, feature_columns).model_fn, model_dir=model_dir, config=config)

    def _predict_one_key(self, input_fn, predict_key):
        for result in self.predict(input_fn=input_fn, predict_keys=[predict_key]):
            yield result[predict_key]

    def predict_cluster_index(self, input_fn):
        """Finds the index of the closest cluster center to each input point.

    Args:
      input_fn: Input points. See `tf.estimator.Estimator.predict`.

    Yields:
      The index of the closest cluster center for each input point.
    """
        for index in self._predict_one_key(input_fn, KMeansClustering.CLUSTER_INDEX):
            yield index

    def score(self, input_fn):
        """Returns the sum of squared distances to nearest clusters.

    Note that this function is different from the corresponding one in sklearn
    which returns the negative sum.

    Args:
      input_fn: Input points. See `tf.estimator.Estimator.evaluate`. Only one
        batch is retrieved.

    Returns:
      The sum of the squared distance from each point in the first batch of
      inputs to its nearest cluster center.
    """
        return self.evaluate(input_fn=input_fn, steps=1)[KMeansClustering.SCORE]

    def transform(self, input_fn):
        """Transforms each input point to its distances to all cluster centers.

    Note that if `distance_metric=KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE`,
    this
    function returns the squared Euclidean distance while the corresponding
    sklearn function returns the Euclidean distance.

    Args:
      input_fn: Input points. See `tf.estimator.Estimator.predict`.

    Yields:
      The distances from each input point to each cluster center.
    """
        for distances in self._predict_one_key(input_fn, KMeansClustering.ALL_DISTANCES):
            if self._distance_metric == KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE:
                yield np.sqrt(distances)
            else:
                yield distances

    def cluster_centers(self):
        """Returns the cluster centers."""
        return self.get_variable_value(KMeansClustering.CLUSTER_CENTERS_VAR_NAME)
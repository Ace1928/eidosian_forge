from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_aggregate_stats
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_bucketize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_feature_split as calculate_best_feature_split
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_feature_split_v2 as calculate_best_feature_split_v2
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_gains_per_feature as calculate_best_gains_per_feature
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_center_bias as center_bias
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_create_quantile_stream_resource as create_quantile_stream_resource
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_example_debug_outputs as example_debug_outputs
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_quantile_summaries as make_quantile_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_stats_summary as make_stats_summary
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_predict as predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_add_summaries as quantile_add_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_deserialize as quantile_resource_deserialize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_flush as quantile_flush
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_get_bucket_boundaries as get_bucket_boundaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_handle_op as quantile_resource_handle_op
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_sparse_aggregate_stats
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_sparse_calculate_best_feature_split as sparse_calculate_best_feature_split
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_training_predict as training_predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble as update_ensemble
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble_v2 as update_ensemble_v2
from tensorflow.python.ops.gen_boosted_trees_ops import is_boosted_trees_quantile_stream_resource_initialized as is_quantile_resource_initialized
from tensorflow.python.training import saver
class QuantileAccumulator:
    """SaveableObject implementation for QuantileAccumulator.

     The bucket boundaries are serialized and deserialized from checkpointing.
  """

    def __init__(self, epsilon, num_streams, num_quantiles, name=None, max_elements=None):
        del max_elements
        self._eps = epsilon
        self._num_streams = num_streams
        self._num_quantiles = num_quantiles
        with ops.name_scope(name, 'QuantileAccumulator') as name:
            self._name = name
            self.resource_handle = self._create_resource()
            self._init_op = self._initialize()
            is_initialized_op = self.is_initialized()
        resources.register_resource(self.resource_handle, self._init_op, is_initialized_op)
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, QuantileAccumulatorSaveable(self.resource_handle, self._init_op, self._num_streams, self.resource_handle.name))

    def _create_resource(self):
        return quantile_resource_handle_op(container='', shared_name=self._name, name=self._name)

    def _initialize(self):
        return create_quantile_stream_resource(self.resource_handle, self._eps, self._num_streams)

    @property
    def initializer(self):
        if self._init_op is None:
            self._init_op = self._initialize()
        return self._init_op

    def is_initialized(self):
        return is_quantile_resource_initialized(self.resource_handle)

    def _serialize_to_tensors(self):
        raise NotImplementedError('When the need arises, TF2 compatibility can be added by implementing this method, along with _restore_from_tensors below.')

    def _restore_from_tensors(self, restored_tensors):
        raise NotImplementedError('When the need arises, TF2 compatibility can be added by implementing this method, along with _serialize_to_tensors above.')

    def add_summaries(self, float_columns, example_weights):
        summaries = make_quantile_summaries(float_columns, example_weights, self._eps)
        summary_op = quantile_add_summaries(self.resource_handle, summaries)
        return summary_op

    def flush(self):
        return quantile_flush(self.resource_handle, self._num_quantiles)

    def get_bucket_boundaries(self):
        return get_bucket_boundaries(self.resource_handle, self._num_streams)
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
class QuantileAccumulatorSaveable(saver.BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for QuantileAccumulator."""

    def __init__(self, resource_handle, create_op, num_streams, name):
        self.resource_handle = resource_handle
        self._num_streams = num_streams
        self._create_op = create_op
        bucket_boundaries = get_bucket_boundaries(self.resource_handle, self._num_streams)
        slice_spec = ''
        specs = []

        def make_save_spec(tensor, suffix):
            return saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name + suffix)
        for i in range(self._num_streams):
            specs += [make_save_spec(bucket_boundaries[i], '_bucket_boundaries_' + str(i))]
        super(QuantileAccumulatorSaveable, self).__init__(self.resource_handle, specs, name)

    def restore(self, restored_tensors, unused_tensor_shapes):
        bucket_boundaries = restored_tensors
        with ops.control_dependencies([self._create_op]):
            return quantile_resource_deserialize(self.resource_handle, bucket_boundaries=bucket_boundaries)
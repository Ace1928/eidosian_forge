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
class PruningMode:
    """Class for working with Pruning modes."""
    NO_PRUNING, PRE_PRUNING, POST_PRUNING = range(0, 3)
    _map = {'none': NO_PRUNING, 'pre': PRE_PRUNING, 'post': POST_PRUNING}

    @classmethod
    def from_str(cls, mode):
        if mode in cls._map:
            return cls._map[mode]
        else:
            raise ValueError('pruning_mode mode must be one of: {}. Found: {}'.format(', '.join(sorted(cls._map)), mode))
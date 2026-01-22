from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.tf_export import tf_export
class _AutoShardDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that shards the `Dataset` automatically.

  This dataset takes in an existing dataset and tries to automatically figure
  out how to shard the dataset in a multi-worker scenario using graph rewrites.

  If the AutoShardPolicy is set to FILE, it walks up the dataset graph until
  it finds a reader dataset, then inserts a ShardDataset op before that node
  so that each worker only sees some files.

  If the AutoShardPolicy is set to DATA, it inserts a ShardDataset op at the
  end of the input pipeline, before any terminal PrefetchDataset if there is
  one. Additionally, if there is a RebatchDatasetV2 in the input pipeline, it
  is written to legacy RebatchDataset for correctness reasons, since
  RebatchDatasetV2 is incompatible with data sharding.

  If the AutoShardPolicy is set to AUTO, it tries to do file-based sharding.
  If it cannot find a reader dataset, it falls back to doing data-based
  sharding.

  If the AutoShardPolicy is set to OFF, it does nothing.

  Attributes:
    num_workers: Total number of workers to shard this dataset across.
    index: The current worker index (out of the total number of workers) this
      dataset is for.
    num_replicas: The total number of replicas across all workers. This is used
      only when sharding by data (either DATA or AUTO) in order to rewrite
      RebatchDatasetV2 to RebatchDataset.

  Raises:
    NotFoundError: If we cannot find a suitable reader dataset to begin
      automatically sharding the dataset.
  """

    def __init__(self, input_dataset, num_workers, index, num_replicas=None):
        self._input_dataset = input_dataset
        self._element_spec = input_dataset.element_spec
        variant_tensor = ged_ops.auto_shard_dataset(self._input_dataset._variant_tensor, num_workers=num_workers, index=index, auto_shard_policy=int(input_dataset.options().experimental_distribute.auto_shard_policy), num_replicas=num_replicas, **self._flat_structure)
        super(_AutoShardDataset, self).__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec
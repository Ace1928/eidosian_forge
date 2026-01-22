from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
class _ShardDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` for sharding its input."""

    def __init__(self, input_dataset, num_shards, index, name):
        """See `Dataset.shard()` for details."""
        self._input_dataset = input_dataset
        self._num_shards = ops.convert_to_tensor(num_shards, dtype=dtypes.int64, name='num_shards')
        self._index = ops.convert_to_tensor(index, dtype=dtypes.int64, name='index')
        self._name = name
        variant_tensor = gen_dataset_ops.shard_dataset(input_dataset._variant_tensor, num_shards=self._num_shards, index=self._index, **self._common_args)
        super().__init__(input_dataset, variant_tensor)
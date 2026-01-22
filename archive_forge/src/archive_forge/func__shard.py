from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
def _shard(input_dataset, num_shards, index, name):
    """See `Dataset.shard()` for details."""
    return _ShardDataset(input_dataset, num_shards, index, name)
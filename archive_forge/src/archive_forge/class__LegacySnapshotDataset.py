from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class _LegacySnapshotDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A Dataset that captures a snapshot or reads from a snapshot."""

    def __init__(self, input_dataset, path, compression=None, reader_path_prefix=None, writer_path_prefix=None, shard_size_bytes=None, pending_snapshot_expiry_seconds=None, num_reader_threads=None, reader_buffer_size=None, num_writer_threads=None, writer_buffer_size=None, shuffle_on_read=None, shuffle_seed=None, mode=None, snapshot_name=None):
        self._compression = compression if compression is not None else ''
        self._reader_path_prefix = reader_path_prefix if reader_path_prefix is not None else ''
        self._writer_path_prefix = writer_path_prefix if writer_path_prefix is not None else ''
        self._shard_size_bytes = shard_size_bytes if shard_size_bytes is not None else -1
        self._pending_snapshot_expiry_seconds = pending_snapshot_expiry_seconds if pending_snapshot_expiry_seconds is not None else -1
        self._num_reader_threads = num_reader_threads if num_reader_threads is not None else -1
        self._reader_buffer_size = reader_buffer_size if reader_buffer_size is not None else -1
        self._num_writer_threads = num_writer_threads if num_writer_threads is not None else -1
        self._writer_buffer_size = writer_buffer_size if writer_buffer_size is not None else -1
        self._shuffle_on_read = shuffle_on_read if shuffle_on_read is not None else False
        self._mode = mode if mode is not None else 'auto'
        self._snapshot_name = snapshot_name if snapshot_name is not None else ''
        self._seed, self._seed2 = random_seed.get_seed(shuffle_seed)
        self._input_dataset = input_dataset
        self._path = ops.convert_to_tensor(path, dtype=dtypes.string, name='path')
        variant_tensor = ged_ops.snapshot_dataset(self._input_dataset._variant_tensor, path=self._path, compression=self._compression, reader_path_prefix=self._reader_path_prefix, writer_path_prefix=self._writer_path_prefix, shard_size_bytes=self._shard_size_bytes, pending_snapshot_expiry_seconds=self._pending_snapshot_expiry_seconds, num_reader_threads=self._num_reader_threads, reader_buffer_size=self._reader_buffer_size, num_writer_threads=self._num_writer_threads, writer_buffer_size=self._writer_buffer_size, shuffle_on_read=self._shuffle_on_read, seed=self._seed, seed2=self._seed2, mode=self._mode, snapshot_name=self._snapshot_name, **self._flat_structure)
        super(_LegacySnapshotDataset, self).__init__(input_dataset, variant_tensor)
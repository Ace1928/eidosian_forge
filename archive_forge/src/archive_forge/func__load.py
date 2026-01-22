import multiprocessing
import os
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import nested_structure_coder
def _load(path, element_spec, compression, reader_func):
    """Loads dataset from tf.data snapshot."""

    def _get_distributed_snapshot_metadata():
        """Reads the distributed snapshot metadata.

    Returns:
      DistributedSnapshotMetadata if the snapshot is a distributed snapshot.
      Returns None if it is a non-distributed snapshot.
    """
        try:
            with gfile.GFile(_pywrap_snapshot_utils.TF_DATA_SnapshotMetadataFilePath(path), 'r') as f:
                return text_format.ParseLines(f, snapshot_pb2.DistributedSnapshotMetadata())
        except (text_format.ParseError, message.DecodeError, UnicodeDecodeError):
            return None
    if reader_func is None:
        reader_func = lambda datasets: datasets.interleave(lambda x: x, cycle_length=multiprocessing.cpu_count(), num_parallel_calls=dataset_ops.AUTOTUNE)
    if element_spec is None:
        with gfile.GFile(os.path.join(path, dataset_ops.DATASET_SPEC_FILENAME), 'rb') as f:
            encoded_spec = f.read()
        element_spec = _parse_element_spec(encoded_spec)
    distributed_snapshot_metadata = _get_distributed_snapshot_metadata()
    if distributed_snapshot_metadata:
        _validate_snapshot(path, distributed_snapshot_metadata, element_spec, compression)
        return _load_distributed_snapshot(path, distributed_snapshot_metadata, reader_func)
    return _LoadDataset(path, element_spec, compression, reader_func)
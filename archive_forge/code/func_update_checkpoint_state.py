import collections
import os.path
import re
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(date=None, instructions='Use `tf.train.CheckpointManager` to manage checkpoints rather than manually editing the Checkpoint proto.')
@tf_export(v1=['train.update_checkpoint_state'])
def update_checkpoint_state(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, latest_filename=None, all_model_checkpoint_timestamps=None, last_preserved_timestamp=None):
    """Updates the content of the 'checkpoint' file.

  This updates the checkpoint file containing a CheckpointState
  proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.
    all_model_checkpoint_timestamps: Optional list of timestamps (floats,
      seconds since the Epoch) indicating when the checkpoints in
      `all_model_checkpoint_paths` were created.
    last_preserved_timestamp: A float, indicating the number of seconds since
      the Epoch when the last preserved checkpoint was written, e.g. due to a
      `keep_checkpoint_every_n_hours` parameter (see
      `tf.train.CheckpointManager` for an implementation).
  Raises:
    RuntimeError: If any of the model checkpoint paths conflict with the file
      containing CheckpointSate.
  """
    update_checkpoint_state_internal(save_dir=save_dir, model_checkpoint_path=model_checkpoint_path, all_model_checkpoint_paths=all_model_checkpoint_paths, latest_filename=latest_filename, save_relative_paths=False, all_model_checkpoint_timestamps=all_model_checkpoint_timestamps, last_preserved_timestamp=last_preserved_timestamp)
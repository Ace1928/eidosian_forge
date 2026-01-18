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
@tf_export('__internal__.train.update_checkpoint_state', v1=[])
def update_checkpoint_state_internal(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, latest_filename=None, save_relative_paths=False, all_model_checkpoint_timestamps=None, last_preserved_timestamp=None):
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
    save_relative_paths: If `True`, will write relative paths to the checkpoint
      state file.
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
    coord_checkpoint_filename = _GetCheckpointFilename(save_dir, latest_filename)
    if save_relative_paths:
        if os.path.isabs(model_checkpoint_path):
            rel_model_checkpoint_path = os.path.relpath(model_checkpoint_path, save_dir)
        else:
            rel_model_checkpoint_path = model_checkpoint_path
        rel_all_model_checkpoint_paths = []
        for p in all_model_checkpoint_paths:
            if os.path.isabs(p):
                rel_all_model_checkpoint_paths.append(os.path.relpath(p, save_dir))
            else:
                rel_all_model_checkpoint_paths.append(p)
        ckpt = generate_checkpoint_state_proto(save_dir, rel_model_checkpoint_path, all_model_checkpoint_paths=rel_all_model_checkpoint_paths, all_model_checkpoint_timestamps=all_model_checkpoint_timestamps, last_preserved_timestamp=last_preserved_timestamp)
    else:
        ckpt = generate_checkpoint_state_proto(save_dir, model_checkpoint_path, all_model_checkpoint_paths=all_model_checkpoint_paths, all_model_checkpoint_timestamps=all_model_checkpoint_timestamps, last_preserved_timestamp=last_preserved_timestamp)
    if coord_checkpoint_filename == ckpt.model_checkpoint_path:
        raise RuntimeError("Save path '%s' conflicts with path used for checkpoint state.  Please use a different save path." % model_checkpoint_path)
    file_io.atomic_write_string_to_file(coord_checkpoint_filename, text_format.MessageToString(ckpt))
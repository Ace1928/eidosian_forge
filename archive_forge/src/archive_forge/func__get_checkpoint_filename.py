from collections import abc
import os
import time
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export
def _get_checkpoint_filename(ckpt_dir_or_file):
    """Returns checkpoint filename given directory or specific checkpoint file."""
    if isinstance(ckpt_dir_or_file, os.PathLike):
        ckpt_dir_or_file = os.fspath(ckpt_dir_or_file)
    if gfile.IsDirectory(ckpt_dir_or_file):
        return checkpoint_management.latest_checkpoint(ckpt_dir_or_file)
    return ckpt_dir_or_file
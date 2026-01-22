import collections
import glob
import os.path
import threading
import time
import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _AddRestoreOps(self, filename_tensor, saveables, restore_sequentially, reshape, preferred_shard=-1, name='restore_all'):
    """Add operations to restore saveables.

    Args:
      filename_tensor: Tensor for the path of the file to load.
      saveables: A list of SaveableObject objects.
      restore_sequentially: True if we want to restore variables sequentially
        within a shard.
      reshape: True if we want to reshape loaded tensors to the shape of the
        corresponding variable.
      preferred_shard: Shard to open first when loading a sharded file.
      name: Name for the returned op.

    Returns:
      An Operation that restores the variables.
    """
    all_tensors = self.bulk_restore(filename_tensor, saveables, preferred_shard, restore_sequentially)
    assign_ops = []
    idx = 0
    for saveable in saveables:
        shapes = None
        if reshape:
            shapes = []
            for spec in saveable.specs:
                v = spec.tensor
                shape = v.get_shape()
                if not shape.is_fully_defined():
                    shape = array_ops.shape(v)
                shapes.append(shape)
        saveable_tensors = all_tensors[idx:idx + len(saveable.specs)]
        idx += len(saveable.specs)
        assign_ops.append(saveable.restore(saveable_tensors, shapes))
    return control_flow_ops.group(*assign_ops, name=name)
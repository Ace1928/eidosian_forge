from typing import Dict, List, Optional
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import save_restore
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _save_cached_when_graph_building(self, file_prefix, object_graph_tensor, options, update_ckpt_state=False):
    """Create or retrieve save ops, overrides parents's private method.

    Args:
      file_prefix: The prefix for saved checkpoint files.
      object_graph_tensor: A `Tensor` to which the current object graph will be
        fed.
      options: `CheckpointOptions` object.
      update_ckpt_state: Optional bool flag. Indiciate whether the internal
        checkpoint state needs to be updated. This is used for async checkpoint,
        which DTrackableSaver currently does not support.
    TODO(chienchunh): Implement async checkpoint for DTrackableSaver.

    Returns:
      A two-element tuple with a filename tensor and a feed_dict of tensors to
      feed when running it (if graph building). The feed dict contains the
      current object graph and any Python state to be saved in the
      checkpoint. When executing eagerly only the first argument is meaningful.
    """
    named_saveable_objects, graph_proto, feed_additions, unused_registered_savers = self._gather_saveables(object_graph_tensor=object_graph_tensor)
    if self._last_save_object_graph != graph_proto or context.executing_eagerly() or ops.inside_function():
        saver = _DSaver(self._mesh, named_saveable_objects)
        save_op = saver.save(file_prefix, options=options)
        with ops.device('/cpu:0'):
            with ops.control_dependencies([save_op]):
                self._cached_save_operation = array_ops.identity(file_prefix)
        self._last_save_object_graph = graph_proto
    return (self._cached_save_operation, feed_additions)
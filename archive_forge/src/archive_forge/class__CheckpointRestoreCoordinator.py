import abc
import collections
import functools
import glob
import os
import threading
import time
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import async_checkpoint_helper
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.checkpoint import save_util
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
class _CheckpointRestoreCoordinator:
    """Holds the status of an object-based checkpoint load."""

    def __init__(self, object_graph_proto, save_path, save_path_tensor, reader, restore_op_cache, graph_view, options, saveables_cache):
        """Specify the checkpoint being loaded.

    Args:
      object_graph_proto: The TrackableObjectGraph protocol buffer associated
        with this checkpoint.
      save_path: A string, the path to the checkpoint, as returned by
        `tf.train.latest_checkpoint`.
      save_path_tensor: A string `Tensor` which contains or will be fed the save
        path.
      reader: A `CheckpointReader` for `save_path`. If None,
        `_CheckpointRestoreCoordinator` will initialize one itself.
      restore_op_cache: A dictionary shared between
        `_CheckpointRestoreCoordinator`s for the same Python objects, used to
        look up restore ops by name to avoid re-creating them across multiple
        `restore()` calls.
      graph_view: A graph_view_lib.ObjectGraphView object for the restored
        objects.
      options: A CheckpointOptions object.
      saveables_cache: An optional cache storing previously created
        SaveableObjects created for each Trackable. Maps Trackables to a
        dictionary of attribute names to Trackable.
    """
        self.options = options
        self.object_graph_proto = object_graph_proto
        self.restore_uid = ops.uid()
        self.unused_attributes = {}
        self.object_by_proto_id = weakref.WeakValueDictionary()
        self.matched_proto_ids = set()
        self.all_python_objects = object_identity.ObjectIdentityWeakSet()
        self.save_path_tensor = save_path_tensor
        self.save_path_string = save_path
        self.dtype_map = reader.get_variable_to_dtype_map()
        self.shape_map = reader.get_variable_to_shape_map()
        self.restore_ops = []
        self.restore_ops_by_name = restore_op_cache
        self.graph_view = graph_view
        self.new_restore_ops_callback = None
        self.deferred_slot_restorations = {}
        self.slot_restorations = {}
        self.expect_partial_attr = False
        for node_index, node in enumerate(self.object_graph_proto.nodes):
            for slot_reference in node.slot_variables:
                self.slot_restorations.setdefault(slot_reference.original_variable_node_id, []).append(base._SlotVariableRestoration(optimizer_id=node_index, slot_variable_id=slot_reference.slot_variable_node_id, slot_name=slot_reference.slot_name))
        self._deleter = _CheckpointRestoreCoordinatorDeleter(self.expect_partial_attr, self.object_graph_proto, self.matched_proto_ids, self.unused_attributes)
        self.saveables_cache = saveables_cache

    @property
    def expect_partial(self):
        return self.expect_partial_attr

    @expect_partial.setter
    def expect_partial(self, expect_partial):
        self.expect_partial_attr = expect_partial
        self._deleter.set_expect_partial(expect_partial)

    def new_restore_ops(self, new_ops):
        self.restore_ops.extend(new_ops)
        if self.new_restore_ops_callback:
            self.new_restore_ops_callback(new_ops)

    def restore_saveables(self, tensor_saveables, python_positions, registered_savers=None, reader=None):
        """Run or build restore operations for SaveableObjects.

    Args:
      tensor_saveables: `SaveableObject`s which correspond to Tensors.
      python_positions: List of CheckpointPositions bound to `PythonState`
        objects which must be restored eagerly.
      registered_savers: a dict mapping saver names-> object name -> Trackable.
      reader: A `CheckpointReader`. If None, a new instance will be created.

    Returns:
      When graph building, a list of restore operations, either cached or newly
      created, to restore `tensor_saveables`.
    """
        if reader is None:
            reader = py_checkpoint_reader.NewCheckpointReader(self.save_path_string)
        restore_ops = []
        for position in python_positions:
            key = position.object_proto.attributes[0].checkpoint_key
            position.trackable.deserialize(reader.get_tensor(key))
        if tensor_saveables or registered_savers:
            flat_saveables = saveable_object_util.validate_and_slice_inputs(tensor_saveables)
            new_restore_ops = functional_saver.MultiDeviceSaver.from_saveables(flat_saveables, registered_savers).restore(self.save_path_tensor, self.options)
            if not context.executing_eagerly():
                for name, restore_op in sorted(new_restore_ops.items()):
                    restore_ops.append(restore_op)
                    assert name not in self.restore_ops_by_name
                    self.restore_ops_by_name[name] = restore_op
        return restore_ops
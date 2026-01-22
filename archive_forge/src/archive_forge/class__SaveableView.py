import collections
import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple
from absl import logging
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_fn
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import pywrap_saved_model
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import tracing_utils
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import trace_saveable_util
from tensorflow.python.types import core as types_core
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
class _SaveableView(object):
    """Provides a frozen view over a trackable root.

  This class helps to create a single stable view over an object to save. The
  saving code should access properties and functions via this class and not via
  the original object as there are cases where an object construct their
  trackable attributes and functions dynamically per call and will yield
  different objects if invoked more than once.

  Changes to the graph, for example adding objects, must happen in
  `augmented_graph_view` (an `_AugmentedGraphView`) before the `_SaveableView`
  is constructed. Changes after the `_SaveableView` has been constructed will be
  ignored.
  """

    def __init__(self, augmented_graph_view: _AugmentedGraphView, options: save_options.SaveOptions):
        """Initializes a SaveableView.

    Args:
      augmented_graph_view: A GraphView object.
      options: A SaveOptions instance.
    """
        self.augmented_graph_view = augmented_graph_view
        self._options = options
        self._trackable_objects, self.node_paths, self.node_ids, self._slot_variables, self.object_names = checkpoint_util.objects_ids_and_slot_variables_and_paths(self.augmented_graph_view)
        untraced_functions = self.augmented_graph_view.untraced_functions
        if untraced_functions:
            logging.info('Found untraced functions such as %s while saving (showing %d of %d). These functions will not be directly callable after loading.', ', '.join(untraced_functions[:_NUM_DISPLAY_UNTRACED_FUNCTIONS]), min(_NUM_DISPLAY_UNTRACED_FUNCTIONS, len(untraced_functions)), len(untraced_functions))
        self._initialize_save_and_restore_functions()
        self._initialize_nodes_and_concrete_functions()
        self.captured_tensor_node_ids = object_identity.ObjectIdentityDictionary()

    def _initialize_save_and_restore_functions(self):
        """Generates all checkpoint save/restore functions.

    The save and restore functions are generated in the eager context (or in the
    user's Graph/Session) before being copied to the exported GraphDef. These
    functions record the ops for saving/restoring the entire object or
    individual objects (e.g. variables and hash tables).

    The global save and restore functions are generated for compatibility with
    TF1 and loading from C++, and is saved in the `MetaGraphDef.saver_def`.

    The individual functions are generated for the Python TF2 use case, where
    users use the loaded SavedModel as-is, or compose new models using parts
    of the object loaded from the SavedModel. These functions are recorded in
    the `saveable_objects` map in the `SavedObject` proto.
    """
        checkpoint_factory_map, registered_savers = save_util_v1.get_checkpoint_factories_and_keys(self.object_names)
        self._obj_to_registered_saver = object_identity.ObjectIdentityDictionary()
        for saver_name, trackables in registered_savers.items():
            for trackable in trackables.values():
                self._obj_to_registered_saver[trackable] = saver_name
        self._saveable_objects_map = _gen_save_and_restore_functions(checkpoint_factory_map)

    def _initialize_nodes_and_concrete_functions(self):
        """Creates graph with nodes for trackable objects and functions.

    Adds functions for each trackable object to `self.nodes` and associated
    concrete functions to `self.concrete_functions` for serialization.
    """
        self.nodes = list(self._trackable_objects)
        self.gradient_functions = []
        self.gradient_defs = []
        for obj in self.nodes:
            if obj in self._saveable_objects_map:
                for save_fn, restore_fn in self._saveable_objects_map[obj].values():
                    self.node_ids[save_fn] = len(self.nodes)
                    self.nodes.append(save_fn)
                    self.node_ids[restore_fn] = len(self.nodes)
                    self.nodes.append(restore_fn)
        self.concrete_functions = [obj for obj in self.nodes if isinstance(obj, defun.ConcreteFunction)]

    @property
    def concrete_and_gradient_functions(self):
        return self.concrete_functions + self.gradient_functions

    @property
    def root(self):
        return self.nodes[0]

    def fill_object_graph_proto(self, proto: saved_object_graph_pb2.SavedObjectGraph):
        """Populate the nodes, children and slot_variables of a SavedObjectGraph."""
        for node_id, node in enumerate(self.nodes):
            assert self.node_ids[node] == node_id
            object_proto = proto.nodes.add()
            object_proto.slot_variables.extend(self._slot_variables.get(node, ()))
            if isinstance(node, _CapturedTensor):
                continue
            for child in self.augmented_graph_view.list_children(node):
                child_proto = object_proto.children.add()
                child_proto.node_id = self.node_ids[child.ref]
                child_proto.local_name = child.name
            for name, ref in self.augmented_graph_view.list_dependencies(node):
                child_proto = object_proto.dependencies.add()
                child_proto.node_id = self.node_ids[ref]
                child_proto.local_name = name
            if node in self._saveable_objects_map:
                assert node not in self._obj_to_registered_saver, "Objects can't have both SaveableObjects and a registered saver"
                for local_name, (save_fn, restore_fn) in self._saveable_objects_map[node].items():
                    saveable_object_proto = object_proto.saveable_objects[local_name]
                    saveable_object_proto.save_function = self.node_ids[save_fn]
                    saveable_object_proto.restore_function = self.node_ids[restore_fn]
            elif node in self._obj_to_registered_saver:
                object_proto.registered_saver = self._obj_to_registered_saver[node]

    def map_resources(self):
        """Makes new resource handle ops corresponding to existing resource tensors.

    Creates resource handle ops in the current default graph, whereas
    `accessible_objects` will be from an eager context. Resource mapping adds
    resource handle ops to the main GraphDef of a SavedModel, which allows the
    C++ loader API to interact with resources.

    Returns:
      A tuple of (object_map, tensor_map, asset_info):
        object_map: A dictionary mapping from object in `accessible_objects` to
          replacement objects created to hold the new resource tensors.
        tensor_map: A dictionary mapping from resource tensors extracted from
          `accessible_objects` to newly created resource tensors.
        asset_info: An _AssetInfo tuple describing external assets referenced
          from accessible_objects.
    """
        assert not context.executing_eagerly()
        object_map = object_identity.ObjectIdentityDictionary()
        tensor_map = object_identity.ObjectIdentityDictionary()
        asset_info = _AssetInfo(asset_defs=[], asset_initializers_by_resource=object_identity.ObjectIdentityDictionary(), asset_filename_map={}, asset_index={})
        for node_id in _dependency_sorted_node_ids(self):
            obj = self.nodes[node_id]
            tensors = obj._export_to_saved_model_graph(object_map=object_map, tensor_map=tensor_map, options=self._options)
            if isinstance(obj, asset.Asset):
                _add_asset_info(obj, asset_info, tensor_map[obj.asset_path])
            if tensors:
                for tensor in tensors:
                    self.captured_tensor_node_ids[tensor] = node_id
        return (object_map, tensor_map, asset_info)

    def add_capture_and_node(self, capture, node):
        node_id = len(self.nodes)
        self.nodes.append(node)
        self.node_ids[capture] = node_id
        self.node_ids[node] = node_id
        self.captured_tensor_node_ids[capture] = node_id
        return node_id

    def get_concrete_resource_initializers(self):
        concrete_initializers = []
        for obj in self.nodes:
            if isinstance(obj, resource.CapturableResource):
                concrete_initializers.append(self.augmented_graph_view.get_child(obj, '_initialize').get_concrete_function())
        return concrete_initializers
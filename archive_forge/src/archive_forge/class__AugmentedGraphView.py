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
class _AugmentedGraphView(graph_view.ObjectGraphView):
    """An extendable graph which also tracks functions attached to objects.

  Extensions through `add_object` appear in the object graph and any checkpoints
  generated from it, even if they are not dependencies of the node they were
  attached to in the saving program. For example a `.signatures` attribute is
  added to exported SavedModel root objects without modifying the root object
  itself.

  Also tracks functions attached to objects in the graph, through the caching
  `_list_functions` method. Enumerating functions only through this method
  ensures that we get a consistent view of functions, even if object attributes
  create new functions every time they are accessed.
  """

    def __init__(self, root):
        super(_AugmentedGraphView, self).__init__(root)
        self._children_cache = object_identity.ObjectIdentityDictionary()
        self._serialization_cache = object_identity.ObjectIdentityDictionary()
        self._wrapped_functions = {}
        self.untraced_functions = []

    def set_signature(self, signature_map: signature_serialization._SignatureMap, wrapped_functions: Dict[Callable[..., Any], Callable[..., Any]]):
        """Attach signature to the root object.

    Args:
      signature_map: An object that contains signature functions.
      wrapped_functions: A dictionary mapping functions to functions that are
        guaranteed to not capture cached variables (functions that capture
        cached variables can't be saved).
    """
        self.list_children(self.root)
        name = signature_serialization.SIGNATURE_ATTRIBUTE_NAME
        self._children_cache[self.root][name] = signature_map
        self._wrapped_functions.update(wrapped_functions)

    def _breadth_first_traversal(self):
        """Returns all trackable objects in the SavedObjectGraph."""
        trackable_objects, _ = super(_AugmentedGraphView, self)._breadth_first_traversal()
        asset_paths = object_identity.ObjectIdentityDictionary()
        constant_captures = object_identity.ObjectIdentityDictionary()
        for obj in trackable_objects:
            if isinstance(obj, asset.Asset):
                asset_paths[obj.asset_path] = obj
            if isinstance(obj, saved_model_utils.TrackableConstant):
                constant_captures[obj.capture] = obj

        def _get_merged_trackable(x):
            if isinstance(x, asset.Asset):
                return asset_paths[x.asset_path]
            if isinstance(x, saved_model_utils.TrackableConstant):
                if x.capture in asset_paths:
                    return asset_paths[x.capture]
                else:
                    return constant_captures[x.capture]
            return x
        for obj in list(self._children_cache.keys()):
            if _get_merged_trackable(obj) is not obj:
                del self._children_cache[obj]
                continue
            for name, child in self._children_cache[obj].items():
                self._children_cache[obj][name] = _get_merged_trackable(child)
        return super(_AugmentedGraphView, self)._breadth_first_traversal()

    def list_children(self, obj):
        """Lists children of `obj` for SavedModel."""
        if obj not in self._children_cache:
            children = self._children_cache[obj] = {}
            for name, child in super(_AugmentedGraphView, self).list_children(obj, save_type=base.SaveType.SAVEDMODEL, cache=self._serialization_cache):
                if isinstance(child, defun.ConcreteFunction):
                    child = self._maybe_uncache_variable_captures(child)
                children[name] = child
            if isinstance(obj, def_function.Function) and (not children):
                self.untraced_functions.append(obj.name)
        for name, child in self._children_cache[obj].items():
            yield base.TrackableReference(name, child)

    def get_child(self, obj, name: str):
        return self._children_cache[obj][name]

    def _maybe_uncache_variable_captures(self, concrete_function: cf.ConcreteFunction):
        if concrete_function in self._wrapped_functions:
            return self._wrapped_functions[concrete_function]
        for capture in concrete_function.captured_inputs:
            if hasattr(capture, '_cached_variable'):
                if concrete_function not in self._wrapped_functions:
                    wrapped = self._wrapped_functions[concrete_function] = function_serialization.wrap_cached_variables(concrete_function)
                    return wrapped
        return concrete_function

    def list_dependencies(self, obj):
        """Yields `Trackables` that must be loaded before `obj`.

    Dependencies and children are both dictionaries of `Trackables`. Children
    define the object graph structure (used in both checkpoints and SavedModel),
    while dependency defines the order used to load the SavedModel

    Args:
      obj: A `Trackable` object

    Yields:
      Tuple of dependency names and trackable objects.

    Raises:
      TypeError: if any of the returned dependencies are not instances of
        `Trackable`.
    """
        if obj not in self._children_cache:
            children = {}
        else:
            children = self._children_cache[obj]
        for name, dep in obj._deserialization_dependencies(children).items():
            if not isinstance(dep, base.Trackable):
                raise TypeError(f"The dependency of type {type(dep)} is not an instance `Trackable`, and can't be saved to SavedModel. Please check the implementation of `_deserialization_dependencies` in the parent object {obj}.")
            yield (name, dep)
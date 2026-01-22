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
def _build_meta_graph_impl(obj, signatures, options: save_options.SaveOptions, meta_graph_def=None):
    """Creates a MetaGraph containing the resources and functions of an object."""
    if ops.inside_function():
        raise AssertionError('`tf.saved_model.save` is not supported inside a traced @tf.function. Move the call to the outer eagerly-executed context.')
    if not isinstance(obj, base.Trackable):
        raise ValueError(f'Expected an object of type `Trackable`, such as `tf.Module` or a subclass of the `Trackable` class, for export. Got {obj} with type {type(obj)}.')
    meta_graph_def = meta_graph_def or meta_graph_pb2.MetaGraphDef()
    augmented_graph_view = _AugmentedGraphView(obj)
    if signatures is None:
        signatures = signature_serialization.find_function_to_export(augmented_graph_view)
    signatures, wrapped_functions, defaults = signature_serialization.canonicalize_signatures(signatures)
    signature_serialization.validate_augmented_graph_view(augmented_graph_view)
    signature_map = signature_serialization.create_signature_map(signatures)
    augmented_graph_view.set_signature(signature_map, wrapped_functions)
    saveable_view = _SaveableView(augmented_graph_view, options)
    object_saver = checkpoint.TrackableSaver(augmented_graph_view)
    asset_info, exported_graph = _fill_meta_graph_def(meta_graph_def, saveable_view, signatures, options.namespace_whitelist, options.experimental_custom_gradients, defaults)
    if options.function_aliases:
        function_aliases = meta_graph_def.meta_info_def.function_aliases
        for alias, func in options.function_aliases.items():
            if isinstance(func, types_core.ConcreteFunction):
                function_aliases[func.name] = alias
            elif isinstance(func, polymorphic_function.Function):
                for fdef in func._list_all_concrete_functions():
                    function_aliases[fdef.name] = alias
            elif isinstance(func, collections.abc.Iterable) and all((isinstance(x, types_core.ConcreteFunction) for x in func)):
                for entry in func:
                    function_aliases[entry.name] = alias
            else:
                raise TypeError(f'Unsupported type f{type(func)}. Functions in `function_aliases` should be created by tf.function, or concrete functions, or collections of concrete functions.')
    object_graph_proto = _serialize_object_graph(saveable_view, asset_info.asset_index)
    meta_graph_def.object_graph_def.CopyFrom(object_graph_proto)
    return (meta_graph_def, exported_graph, object_saver, asset_info, saveable_view.nodes, saveable_view.node_paths)
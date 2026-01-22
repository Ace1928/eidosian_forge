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
def _gen_save_and_restore_functions(checkpoint_factory_map: object_identity.ObjectIdentityDictionary) -> object_identity.ObjectIdentityDictionary:
    """Generates global and individual save/restore concrete functions.

  The global functions records the ops to save and restore the entire object to
  a file prefix, while the individual functions save and restore value tensors
  for resources.

  This function is intended to run on the output of
  `save_util_v1.get_checkpoint_factories_and_keys(object_names)`,
  which returns the generated a map of `_CheckpointFactoryData`.

  Args:
    checkpoint_factory_map: A dictionary mapping trackable objects to
      a list of `_CheckpointFactoryData`.

  Returns:
    Tuple of (
      saveable_fn_map: Maps obj -> factory name -> (concrete save, restore)
      )
  """
    saveable_fn_map = object_identity.ObjectIdentityDictionary()
    for obj, factory_data_list in checkpoint_factory_map.items():
        if resource_variable_ops.is_resource_variable(obj) or not factory_data_list:
            continue
        if factory_data_list[0].name == trackable_utils.SERIALIZE_TO_TENSORS_NAME:
            assert len(factory_data_list) == 1
            saveable_fn_map[obj] = {trackable_utils.SERIALIZE_TO_TENSORS_NAME: tracing_utils.trace_save_and_restore(obj)}
        else:
            saveable_fn_map[obj] = trace_saveable_util.trace_save_restore_function_map(obj, factory_data_list)
    return saveable_fn_map
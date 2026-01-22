import abc
import functools
import queue
import threading
import warnings
import numpy as np
from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as tracking_base
from tensorflow.python.trackable import resource as resource_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _trace_variant_creation(self):
    """Traces a function which outputs a variant `tf.Tensor` for this dataset.

    Note that creating this function involves evaluating an op, and is currently
    only supported when executing eagerly.

    Returns:
      A zero-argument `ConcreteFunction` which outputs a variant `tf.Tensor`.
    """
    variant = self._variant_tensor
    if not isinstance(variant, ops.EagerTensor):
        raise NotImplementedError('Constructing a tf.function that reproduces a given dataset is only supported for datasets created eagerly. Please file a feature request if this is important to you.')
    with context.eager_mode(), ops.device('CPU'):
        graph_def = graph_pb2.GraphDef().FromString(self._as_serialized_graph(external_state_policy=options_lib.ExternalStatePolicy.FAIL).numpy())
    output_node_names = []
    for node in graph_def.node:
        if node.op == '_Retval':
            output_node_names = node.input
    if len(output_node_names) != 1:
        raise AssertionError(f'Dataset graph is expected to only have one return value but found {len(output_node_names)} return values: {output_node_names}.')
    output_node_name = output_node_names[0]
    file_path_nodes = {}
    if ops.get_default_graph().building_function:
        asset_tracker = self._maybe_track_assets(graph_def)
        for key in asset_tracker:
            assets_list = [array_ops.expand_dims(asset.asset_path, axis=0) for asset in asset_tracker[key]]
            file_path_nodes[key] = array_ops.concat(assets_list, axis=0)
    variant_function = wrap_function.function_from_graph_def(graph_def, inputs=[], outputs=output_node_name + ':0', captures=file_path_nodes)
    for used_function in self._functions():
        used_function.function.add_to_graph(variant_function.graph)
    return variant_function
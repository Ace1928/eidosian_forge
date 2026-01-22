import collections
from functools import partial  # pylint: disable=g-importing-member
import os
import platform
import sys
import tempfile
import numpy as np
import six as _six
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _remove_native_segments(input_func):
    """Remove native segments from the input TF-TRT Converted Function.

  Args:
    input_func: provide the concrete function with native segment nodes. The
      transformed output func will not contain any native segment nodes. All the
      TRTEngineOp references will be deleted and reset to default empty func.
  """
    input_graph_def = input_func.graph.as_graph_def()
    nodes_deleted = 0
    for func_id in reversed(range(len(input_graph_def.library.function))):
        f = input_graph_def.library.function[func_id]
        if 'native_segment' in f.signature.name:
            nodes_deleted += 1
            while context.context().has_function(f.signature.name):
                context.context().remove_function(f.signature.name)
            del input_graph_def.library.function[func_id]
    logging.info(f'Found and deleted native segments from {nodes_deleted} TRTEngineOp nodes.')
    for node in input_graph_def.node:
        if node.op == 'TRTEngineOp':
            del node.attr['segment_func']
    for func in input_graph_def.library.function:
        for node in func.node_def:
            if node.op == 'TRTEngineOp':
                del node.attr['segment_func']
    new_func = _construct_function_from_graph_def(input_func, input_graph_def)
    return new_func
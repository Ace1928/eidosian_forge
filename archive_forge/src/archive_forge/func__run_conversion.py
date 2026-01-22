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
def _run_conversion(self, meta_graph_def):
    """Run Grappler's OptimizeGraph() tool to convert the graph.

    Args:
      meta_graph_def: the MetaGraphDef instance to run the optimizations on.

    Returns:
      The optimized GraphDef.
    """
    grappler_session_config = config_pb2.ConfigProto()
    custom_rewriter_config = _get_tensorrt_rewriter_config(conversion_params=self._conversion_params._replace(allow_build_at_runtime=True), is_dynamic_op=True, max_batch_size=None, disable_non_trt_optimizers=self._test_only_disable_non_trt_optimizers, use_implicit_batch=not self._use_dynamic_shape, profile_strategy=self._profile_strategy)
    grappler_session_config.graph_options.rewrite_options.CopyFrom(custom_rewriter_config)
    return tf_optimizer.OptimizeGraph(grappler_session_config, meta_graph_def, graph_id=b'tf_graph')
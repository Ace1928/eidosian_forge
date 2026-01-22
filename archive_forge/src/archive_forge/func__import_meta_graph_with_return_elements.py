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
def _import_meta_graph_with_return_elements(meta_graph_or_file, clear_devices=False, import_scope=None, return_elements=None, **kwargs):
    """Import MetaGraph, and return both a saver and returned elements."""
    if context.executing_eagerly():
        raise RuntimeError('Exporting/importing meta graphs is not supported when eager execution is enabled. No graph exists when eager execution is enabled.')
    if not isinstance(meta_graph_or_file, meta_graph_pb2.MetaGraphDef):
        meta_graph_def = meta_graph.read_meta_graph_file(meta_graph_or_file)
    else:
        meta_graph_def = meta_graph_or_file
    imported_vars, imported_return_elements = meta_graph.import_scoped_meta_graph_with_return_elements(meta_graph_def, clear_devices=clear_devices, import_scope=import_scope, return_elements=return_elements, **kwargs)
    saver = _create_saver_from_imported_meta_graph(meta_graph_def, import_scope, imported_vars)
    return (saver, imported_return_elements)
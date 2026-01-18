import os
import sys
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
Load the MetaGraphDef graph and restore variable values into the session.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
      **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.

    Returns:
      `MetagraphDef` proto of the graph that was loaded.
    
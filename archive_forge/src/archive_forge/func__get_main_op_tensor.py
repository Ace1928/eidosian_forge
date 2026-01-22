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
def _get_main_op_tensor(meta_graph_def_to_load, init_op_key=constants.MAIN_OP_KEY):
    """Gets the main op tensor, if one exists.

  Args:
    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.
    init_op_key: name of the collection to check; should be one of MAIN_OP_KEY
      or the deprecated LEGACY_INIT_OP_KEY

  Returns:
    The main op tensor, if it exists and `None` otherwise.

  Raises:
    RuntimeError: If the collection def corresponding to the main op key has
        other than exactly one tensor.
  """
    collection_def = meta_graph_def_to_load.collection_def
    init_op = None
    if init_op_key in collection_def:
        init_op_list = collection_def[init_op_key].node_list.value
        if len(init_op_list) != 1:
            raise RuntimeError(f'Expected exactly one SavedModel init op. Found {len(init_op_list)}: {init_op_list}.')
        init_op = ops.get_collection(init_op_key)[0]
    return init_op
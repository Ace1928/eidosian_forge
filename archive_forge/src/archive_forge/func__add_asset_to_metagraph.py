import functools
import os
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _add_asset_to_metagraph(meta_graph_def, asset_filename, asset_tensor):
    """Builds an asset proto and adds it to the meta graph def.

  Args:
    meta_graph_def: The meta graph def to which the asset will be added.
    asset_filename: The filename of the asset to be added.
    asset_tensor: The asset tensor used to populate the tensor info of the asset
      proto.
  """
    asset_proto = meta_graph_def.asset_file_def.add()
    asset_proto.filename = asset_filename
    asset_proto.tensor_info.name = asset_tensor.name
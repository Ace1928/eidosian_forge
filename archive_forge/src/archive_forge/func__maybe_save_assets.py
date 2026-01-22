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
def _maybe_save_assets(write_fn, assets_to_add=None):
    """Saves assets to the meta graph.

  Args:
    write_fn: A function callback that writes assets into meta graph.
    assets_to_add: The list where the asset paths are setup.

  Returns:
    A dict of asset basenames for saving to the original full path to the asset.

  Raises:
    ValueError: Indicating an invalid filepath tensor.
  """
    asset_filename_map = {}
    if assets_to_add is None:
        tf_logging.info('No assets to save.')
        return asset_filename_map
    for asset_tensor in assets_to_add:
        asset_source_filepath = _asset_path_from_tensor(asset_tensor)
        if not asset_source_filepath:
            raise ValueError(f'Asset filepath tensor {asset_tensor} in is invalid.')
        asset_filename = get_asset_filename_to_add(asset_source_filepath, asset_filename_map)
        write_fn(asset_filename, asset_tensor)
        asset_filename_map[asset_filename] = asset_source_filepath
    tf_logging.info('Assets added to graph.')
    return asset_filename_map
import argparse
import platform
import ast
import os
import re
from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
import numpy as np
from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import meta_graph as meta_graph_lib
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_aot_compile
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.tpu import tpu
from tensorflow.python.util.compat import collections_abc
def _show_all(saved_model_dir):
    """Prints tag-set, ops, SignatureDef, and Inputs/Outputs of SavedModel.

  Prints all tag-set, ops, SignatureDef and Inputs/Outputs information stored in
  SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
  """
    saved_model = saved_model_utils.read_saved_model(saved_model_dir)
    for meta_graph_def in sorted(saved_model.meta_graphs, key=lambda meta_graph_def: list(meta_graph_def.meta_info_def.tags)):
        tag_set = meta_graph_def.meta_info_def.tags
        print("\nMetaGraphDef with tag-set: '%s' contains the following SignatureDefs:" % ', '.join(tag_set))
        tag_set = ','.join(tag_set)
        signature_def_map = meta_graph_def.signature_def
        for signature_def_key in sorted(signature_def_map.keys()):
            print("\nsignature_def['" + signature_def_key + "']:")
            _show_inputs_outputs_mgd(meta_graph_def, signature_def_key, indent=1)
        _show_ops_in_metagraph_mgd(meta_graph_def)
    _show_defined_functions(saved_model_dir, saved_model.meta_graphs)
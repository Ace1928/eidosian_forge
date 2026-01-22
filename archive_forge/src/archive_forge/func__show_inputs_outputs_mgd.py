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
def _show_inputs_outputs_mgd(meta_graph_def, signature_def_key, indent):
    """Prints input and output TensorInfos.

  Prints the details of input and output TensorInfos for the SignatureDef mapped
  by the given signature_def_key.

  Args:
    meta_graph_def: MetaGraphDef to inspect.
    signature_def_key: A SignatureDef key string.
    indent: How far (in increments of 2 spaces) to indent each line of output.
  """
    inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)
    outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)
    indent_str = '  ' * indent

    def in_print(s):
        print(indent_str + s)
    in_print('The given SavedModel SignatureDef contains the following input(s):')
    for input_key, input_tensor in sorted(inputs_tensor_info.items()):
        in_print("  inputs['%s'] tensor_info:" % input_key)
        _print_tensor_info(input_tensor, indent + 1)
    in_print('The given SavedModel SignatureDef contains the following output(s):')
    for output_key, output_tensor in sorted(outputs_tensor_info.items()):
        in_print("  outputs['%s'] tensor_info:" % output_key)
        _print_tensor_info(output_tensor, indent + 1)
    in_print('Method name is: %s' % meta_graph_def.signature_def[signature_def_key].method_name)
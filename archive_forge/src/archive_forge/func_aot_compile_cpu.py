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
def aot_compile_cpu():
    """Function triggered by aot_compile_cpu command."""
    checkpoint_path = _SMCLI_CHECKPOINT_PATH.value or os.path.join(_SMCLI_DIR.value, 'variables/variables')
    if not _SMCLI_VARIABLES_TO_FEED.value:
        variables_to_feed = []
    elif _SMCLI_VARIABLES_TO_FEED.value.lower() == 'all':
        variables_to_feed = None
    else:
        variables_to_feed = _SMCLI_VARIABLES_TO_FEED.value.split(',')
    saved_model_aot_compile.aot_compile_cpu_meta_graph_def(checkpoint_path=checkpoint_path, meta_graph_def=saved_model_utils.get_meta_graph_def(_SMCLI_DIR.value, _SMCLI_TAG_SET.value), signature_def_key=_SMCLI_SIGNATURE_DEF_KEY.value, variables_to_feed=variables_to_feed, output_prefix=_SMCLI_OUTPUT_PREFIX.value, target_triple=_SMCLI_TARGET_TRIPLE.value, target_cpu=_SMCLI_TARGET_CPU.value, cpp_class=_SMCLI_CPP_CLASS.value, multithreading=_SMCLI_MULTITHREADING.value.lower() not in ('f', 'false', '0'))
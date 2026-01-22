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
def add_aot_compile_cpu_subparser(subparsers):
    """Add parser for `aot_compile_cpu`."""
    compile_msg = '\n'.join(['Usage example:', 'To compile a SavedModel signature via (CPU) XLA AOT:', '$saved_model_cli aot_compile_cpu \\', '   --dir /tmp/saved_model \\', '   --tag_set serve \\', '   --output_dir /tmp/saved_model_xla_aot', '', '', 'Note: Additional XLA compilation options are available by setting the ', 'XLA_FLAGS environment variable.  See the XLA debug options flags for ', 'all the options: ', '  {}'.format(_XLA_DEBUG_OPTIONS_URL), '', 'For example, to disable XLA fast math when compiling:', '', 'XLA_FLAGS="--xla_cpu_enable_fast_math=false" $saved_model_cli ', 'aot_compile_cpu ...', '', 'Some possibly useful flags:', '  --xla_cpu_enable_fast_math=false', '  --xla_force_host_platform_device_count=<num threads>', '    (useful in conjunction with disabling multi threading)'])
    parser_compile = subparsers.add_parser('aot_compile_cpu', description=compile_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser_compile.set_defaults(func=aot_compile_cpu)
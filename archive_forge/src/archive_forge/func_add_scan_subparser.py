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
def add_scan_subparser(subparsers):
    """Add parser for `scan`."""
    scan_msg = 'Usage example:\nTo scan for default denylisted ops in SavedModel:\n$saved_model_cli scan --dir /tmp/saved_model\nTo scan for a specific set of ops in SavedModel:\n$saved_model_cli scan --dir /tmp/saved_model --op_denylist OpName,OpName,OpName\nTo scan a specific MetaGraph, pass in --tag_set\n'
    parser_scan = subparsers.add_parser('scan', description=scan_msg, formatter_class=argparse.RawTextHelpFormatter)
    parser_scan.set_defaults(func=scan)
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
def _print_args(arguments, argument_type='Argument', indent=0):
    """Formats and prints the argument of the concrete functions defined in the model.

  Args:
    arguments: Arguments to format print.
    argument_type: Type of arguments.
    indent: How far (in increments of 2 spaces) to indent each line of
     output.
  """
    indent_str = '  ' * indent

    def _maybe_add_quotes(value):
        is_quotes = "'" * isinstance(value, str)
        return is_quotes + str(value) + is_quotes

    def in_print(s, end='\n'):
        print(indent_str + s, end=end)
    for index, element in enumerate(arguments, 1):
        if indent == 4:
            in_print('%s #%d' % (argument_type, index))
        if isinstance(element, str):
            in_print('  %s' % element)
        elif isinstance(element, tensor_spec.TensorSpec):
            print((indent + 1) * '  ' + '%s: %s' % (element.name, repr(element)))
        elif isinstance(element, collections_abc.Iterable) and (not isinstance(element, dict)):
            in_print('  DType: %s' % type(element).__name__)
            in_print('  Value: [', end='')
            for value in element:
                print('%s' % _maybe_add_quotes(value), end=', ')
            print('\x08\x08]')
        elif isinstance(element, dict):
            in_print('  DType: %s' % type(element).__name__)
            in_print('  Value: {', end='')
            for key, value in element.items():
                print("'%s': %s" % (str(key), _maybe_add_quotes(value)), end=', ')
            print('\x08\x08}')
        else:
            in_print('  DType: %s' % type(element).__name__)
            in_print('  Value: %s' % str(element))
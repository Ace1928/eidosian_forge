import collections
import errno
import gc
import itertools
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
import numpy as np
from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
def _VerifyConnections(self, expected_engines, expected_input_map, original_gdef, converted_gdef):
    """Checks that the converted graph contains the expected connections."""
    old_to_new_node_map = {self._ToString(node.name): self._ToString(node.name) for node in original_gdef.node}
    for engine_name, node_names in expected_engines.items():
        for node_name in node_names:
            old_to_new_node_map[node_name] = engine_name

    def _InputName(inp):
        inp = self._ToString(inp)
        prefix = ''
        if inp[0] == '^':
            prefix = '^'
            inp = inp[1:]
        parts = inp.split(':')
        if len(parts) > 1 and parts[-1].isdigit():
            inp = inp[:-len(parts[-1]) - 1]
        return (prefix, inp)
    new_cast_op_name_to_node_map = {node.name: node for node in converted_gdef.node if node.name not in old_to_new_node_map and node.op == 'Cast'}
    actual_input_map = {}
    for node in converted_gdef.node:
        name_str = node.name
        if node.op == 'TRTEngineOp':
            name_str = self._RemoveGraphSequenceNumber(name_str)
        elif name_str not in old_to_new_node_map:
            continue
        actual_input_map[name_str] = set()
        input_set = actual_input_map[name_str]
        for inp in node.input:
            prefix, node_name = _InputName(inp)
            node_name = self._MayRemoveGraphSequenceNumber(node_name)
            if node_name in new_cast_op_name_to_node_map:
                prefix, node_name = _InputName(new_cast_op_name_to_node_map[node_name].input[0])
            input_set.add(prefix + node_name)
    self.assertEqual(expected_input_map, actual_input_map, msg='\nexpected:\n%s\nvs actual:\n%s' % (sorted(expected_input_map.items()), sorted(actual_input_map.items())))
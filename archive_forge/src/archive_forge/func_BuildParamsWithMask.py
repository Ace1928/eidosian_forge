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
def BuildParamsWithMask(self, graph_fn, dtype, input_shapes, output_shapes, input_mask, output_mask, extra_inputs, extra_outputs):
    """Build test parameters with static or dynamic input shapes.

    To define dynamic shapes give a boolean mask that describes which
    dimensions to treat as known. The values in input_mask are interpreted the
    following way:
    - True: known dim (use the corresponding value from input_shapes)
    - False: unknown dim (replace the corresponding value from input_shapes
             with None)
    For example, to define the first two dimension with unknown size use
    input_shapes=[[1,2,1,8]], input_mask=[[False, False, True, True]].

    Args:
      graph_fn: The function to build the graph.
      dtype: The element type.
      input_shapes: The input shapes.
      output_shapes: The output shapes.
      input_mask: The input shape masks.
      output_mask: the output shape masks.
      extra_inputs: list of additional input shapes
      extra_outputs: list of additional outputs shapes

    Returns:
      The test parameters.
    """

    def _ValidateShapes(shapes):
        for shape in shapes:
            assert all(shape), f'Shape unspecified: {shape}'
    _ValidateShapes(input_shapes)
    _ValidateShapes(output_shapes)
    assert len(input_mask) == len(input_shapes), f'Inconsistent input_mask and input_shapes: len({input_mask}) != len({input_shapes}).'
    assert len(output_mask) == len(output_shapes), f'Inconsistent output_mask and output_shapes: len({output_mask}) != len({output_shapes}).'
    for extra_in_shape, extra_out_shape in zip(extra_inputs, extra_outputs):
        assert len(input_shapes) == len(extra_in_shape), f'Inconsistent input_shapes and extra_in_shape: len({input_shapes}) != len({extra_in_shape}).'
        assert len(output_shapes) == len(extra_out_shape), f'Inconsistent output_shapes and extra_out_shape: len({output_shapes}) != len({extra_out_shape}).'
    return TfTrtIntegrationTestParams(graph_fn=graph_fn, input_specs=[self._GetTensorSpec(shape, mask, dtype, 'input_%d' % i) for i, (shape, mask) in enumerate(zip(input_shapes, input_mask))], output_specs=[self._GetTensorSpec(shape, mask, dtype, 'output_%d' % i) for i, (shape, mask) in enumerate(zip(output_shapes, output_mask))], input_dims=[input_shapes] + extra_inputs, expected_output_dims=[output_shapes] + extra_outputs)
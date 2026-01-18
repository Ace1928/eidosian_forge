import distutils.spawn
import enum
import hashlib
import os as _os
import platform as _platform
import subprocess as _subprocess
import tempfile as _tempfile
from typing import Optional
import warnings
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_options_pb2 as quant_opts_pb2
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import util
from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import ConverterError
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper as _metrics_wrapper
from tensorflow.lite.toco import model_flags_pb2 as _model_flags_pb2
from tensorflow.lite.toco import toco_flags_pb2 as _conversion_flags_pb2
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader as _resource_loader
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export as _tf_export
@_tf_export(v1=['lite.toco_convert'])
@deprecation.deprecated(None, 'Use `lite.TFLiteConverter` instead.')
def toco_convert(input_data, input_tensors, output_tensors, *args, **kwargs):
    """Convert a TensorFlow GraphDef to TFLite.

  This function is deprecated. Please use `tf.lite.TFLiteConverter` API instead.
  Conversion can be customized by providing arguments that are forwarded to
  `build_model_flags` and `build_conversion_flags` (see documentation for
  details).
  Args:
    input_data: Input data (i.e. often `sess.graph_def`).
    input_tensors: List of input tensors. Type and shape are computed using
      `foo.shape` and `foo.dtype`.
    output_tensors: List of output tensors (only .name is used from this).
    *args: See `build_model_flags` and `build_conversion_flags`.
    **kwargs: See `build_model_flags` and `build_conversion_flags`.

  Returns:
    The converted TensorFlow Lite model in a bytes array.

  Raises:
    Defined in `convert`.
  """
    kwargs['enable_mlir_converter'] = kwargs.get('enable_mlir_converter', False)
    return convert_graphdef(input_data, input_tensors, output_tensors, *args, **kwargs)
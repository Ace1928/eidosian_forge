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
@convert_phase(Component.CONVERT_TF_TO_TFLITE_MODEL, SubComponent.CONVERT_GRAPHDEF_USING_DEPRECATED_CONVERTER)
def _run_deprecated_conversion_binary(model_flags_str, conversion_flags_str, input_data_str, debug_info_str=None):
    """Convert `input_data_str` using deprecated conversion binary.

  Args:
    model_flags_str: Serialized proto describing model properties, see
      `model_flags.proto`.
    conversion_flags_str: Serialized proto describing TFLite converter
      properties, see `toco/toco_flags.proto`.
    input_data_str: Input data in serialized form (e.g. a graphdef is common)
    debug_info_str: Serialized `GraphDebugInfo` proto describing logging
      information. (default None)

  Returns:
    Converted model in serialized form (e.g. a TFLITE model is common).
  Raises:
    ConverterError: When cannot find the deprecated conversion binary.
    RuntimeError: When conversion fails, an exception is raised with the error
      message embedded.
  """
    if distutils.spawn.find_executable(_deprecated_conversion_binary) is None:
        raise ConverterError('Could not find `toco_from_protos` binary, make sure\nyour virtualenv bin directory or pip local bin directory is in your path.\nIn particular, if you have installed TensorFlow with --user, make sure you\nadd the install directory to your path.\n\nFor example:\nLinux: export PATH=$PATH:~/.local/bin/\nMac: export PATH=$PATH:~/Library/Python/<version#>/bin\n\nAlternative, use virtualenv.')
    conversion_filename: str = None
    model_filename: str = None
    input_filename: str = None
    output_filename: str = None
    try:
        with _tempfile.NamedTemporaryFile(delete=False) as fp_conversion, _tempfile.NamedTemporaryFile(delete=False) as fp_model, _tempfile.NamedTemporaryFile(delete=False) as fp_input, _tempfile.NamedTemporaryFile(delete=False) as fp_debug:
            conversion_filename = fp_conversion.name
            input_filename = fp_input.name
            model_filename = fp_model.name
            debug_filename = fp_debug.name
            fp_model.write(model_flags_str)
            fp_conversion.write(conversion_flags_str)
            fp_input.write(input_data_str)
            debug_info_str = debug_info_str if debug_info_str else ''
            if not isinstance(debug_info_str, bytes):
                fp_debug.write(debug_info_str.encode('utf-8'))
            else:
                fp_debug.write(debug_info_str)
        with _tempfile.NamedTemporaryFile(delete=False) as fp:
            output_filename = fp.name
        cmd = [_deprecated_conversion_binary, model_filename, conversion_filename, input_filename, output_filename, '--debug_proto_file={}'.format(debug_filename)]
        cmdline = ' '.join(cmd)
        is_windows = _platform.system() == 'Windows'
        proc = _subprocess.Popen(cmdline, shell=True, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT, close_fds=not is_windows)
        stdout, stderr = proc.communicate()
        exitcode = proc.returncode
        if exitcode == 0:
            with open(output_filename, 'rb') as fp:
                return fp.read()
        else:
            stdout = _try_convert_to_unicode(stdout)
            stderr = _try_convert_to_unicode(stderr)
            raise ConverterError('See console for info.\n%s\n%s\n' % (stdout, stderr))
    finally:
        for filename in [conversion_filename, input_filename, model_filename, output_filename]:
            try:
                _os.unlink(filename)
            except (OSError, TypeError):
                pass
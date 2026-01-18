import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def xxd_output_to_object(input_cc_file):
    """Converts xxd output C++ source file to object.

  Args:
    input_cc_file: Full path name to th C++ source file dumped by xxd

  Raises:
    RuntimeError: If input_cc_file path is invalid.
    IOError: If input_cc_file cannot be opened.

  Returns:
    A python object corresponding to the input tflite file.
  """
    model_bytes = xxd_output_to_bytes(input_cc_file)
    return convert_bytearray_to_object(model_bytes)
import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def xxd_output_to_bytes(input_cc_file):
    """Converts xxd output C++ source file to bytes (immutable).

  Args:
    input_cc_file: Full path name to th C++ source file dumped by xxd

  Raises:
    RuntimeError: If input_cc_file path is invalid.
    IOError: If input_cc_file cannot be opened.

  Returns:
    A bytearray corresponding to the input cc file array.
  """
    pattern = re.compile('\\W*(0x[0-9a-fA-F,x ]+).*')
    model_bytearray = bytearray()
    with open(input_cc_file) as file_handle:
        for line in file_handle:
            values_match = pattern.match(line)
            if values_match is None:
                continue
            list_text = values_match.group(1)
            values_text = filter(None, list_text.split(','))
            values = [int(x, base=16) for x in values_text]
            model_bytearray.extend(values)
    return bytes(model_bytearray)
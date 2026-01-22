import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
Calculates the number of unique resource variables in a model.

  Args:
    model: the input tflite model, either as bytearray or object.

  Returns:
    An integer number representing the number of unique resource variables.
  
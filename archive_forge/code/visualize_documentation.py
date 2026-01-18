import json
import os
import re
import sys
import numpy as np
Returns html description with the given tflite model.

  Args:
    tflite_input: TFLite flatbuffer model path or model object.
    input_is_filepath: Tells if tflite_input is a model path or a model object.

  Returns:
    Dump of the given tflite model in HTML format.

  Raises:
    RuntimeError: If the input is not valid.
  
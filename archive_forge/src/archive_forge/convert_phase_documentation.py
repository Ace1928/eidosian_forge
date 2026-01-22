import collections
import enum
import functools
from typing import Text
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics
If the message matches a pattern, assigns the associated error code.

    It is difficult to assign an error code to some errrors in MLIR side, Ex:
    errors thrown by other components than TFLite or not using mlir::emitError.
    This function try to detect them by the error message and assign the
    corresponding error code.

    Args:
      message: The error message of this exception.
    
import functools
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.python.util.tf_export import tf_export as _tf_export
Returns list of compatibility log messages.

    WARNING: This method should only be used for unit tests.

    Returns:
      The list of log messages by the recent compatibility check.
    Raises:
      RuntimeError: when the compatibility was NOT checked.
    
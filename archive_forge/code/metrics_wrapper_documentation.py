from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics._pywrap_tensorflow_lite_metrics_wrapper import MetricsWrapper  # pylint: disable=unused-import
Returns and clears the list of collected errors in ErrorCollector.

  The RetrieveCollectedErrors function in C++ returns a list of serialized proto
  messages. This function will convert them to ConverterErrorData instances.

  Returns:
    A list of ConverterErrorData.
  
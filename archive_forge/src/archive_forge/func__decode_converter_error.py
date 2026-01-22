import functools
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.python.util.tf_export import tf_export as _tf_export
def _decode_converter_error(self, err):
    """Parses the given ConverterError which has detailed error information."""
    custom_ops = []
    custom_ops_location = []
    tf_ops = []
    tf_ops_location = []
    gpu_not_compatible_ops = []
    for err in err.errors:
        if err.error_code == converter_error_data_pb2.ConverterErrorData.ERROR_NEEDS_CUSTOM_OPS:
            custom_ops.append(err.operator.name)
            custom_ops_location.append(err.location)
        elif err.error_code == converter_error_data_pb2.ConverterErrorData.ERROR_NEEDS_FLEX_OPS:
            tf_ops.append(err.operator.name)
            tf_ops_location.append(err.location)
        elif err.error_code == converter_error_data_pb2.ConverterErrorData.ERROR_GPU_NOT_COMPATIBLE:
            gpu_not_compatible_ops.append(err.operator.name)
            self._log(err.error_message.splitlines()[0])
            self._log(self._get_location_string(err.location) + '\n')
        else:
            self._log(f'{_AUTHORING_ERROR_HDR}: {err.error_message}')
            self._log(self._get_location_string(err.location) + '\n')
    if custom_ops:
        custom_ops_str = ', '.join(sorted(custom_ops))
        err_string = f"{_AUTHORING_ERROR_HDR}: op '{custom_ops_str}' is(are) not natively supported by TensorFlow Lite. You need to provide a custom operator. https://www.tensorflow.org/lite/guide/ops_custom"
        self._log(err_string)
        self._dump_error_details(custom_ops, custom_ops_location)
    if tf_ops:
        tf_ops_str = ', '.join(sorted(tf_ops))
        err_string = f"""{_AUTHORING_WARNING_HDR}: op '{tf_ops_str}' require(s) "Select TF Ops" for model conversion for TensorFlow Lite. https://www.tensorflow.org/lite/guide/ops_select"""
        self._log(err_string)
        self._dump_error_details(tf_ops, tf_ops_location)
    if gpu_not_compatible_ops:
        not_compatible_ops_str = ', '.join(sorted(gpu_not_compatible_ops))
        err_string = f"{_AUTHORING_WARNING_HDR}: op '{not_compatible_ops_str}' aren't compatible with TensorFlow Lite GPU delegate. https://www.tensorflow.org/lite/performance/gpu"
        self._log(err_string)
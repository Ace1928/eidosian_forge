import functools
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.python.util.tf_export import tf_export as _tf_export
def _decode_error_legacy(self, err):
    """Parses the given legacy ConverterError for OSS."""
    for line in str(err).splitlines():
        if line.startswith(_CUSTOM_OPS_HDR):
            custom_ops = line[len(_CUSTOM_OPS_HDR):]
            err_string = f"{_AUTHORING_ERROR_HDR}: op '{custom_ops}' is(are) not natively supported by TensorFlow Lite. You need to provide a custom operator. https://www.tensorflow.org/lite/guide/ops_custom"
            self._log(err_string)
        elif line.startswith(_TF_OPS_HDR):
            tf_ops = line[len(_TF_OPS_HDR):]
            err_string = f"""{_AUTHORING_WARNING_HDR}: op '{tf_ops}' require(s) "Select TF Ops" for model conversion for TensorFlow Lite. https://www.tensorflow.org/lite/guide/ops_select"""
            self._log(err_string)
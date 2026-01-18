from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import _pywrap_toco_api
def wrapped_toco_convert(model_flags_str, toco_flags_str, input_data_str, debug_info_str, enable_mlir_converter):
    """Wraps TocoConvert with lazy loader."""
    return _pywrap_toco_api.TocoConvert(model_flags_str, toco_flags_str, input_data_str, False, debug_info_str, enable_mlir_converter)
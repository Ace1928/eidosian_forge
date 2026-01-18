from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import _pywrap_toco_api
def wrapped_experimental_mlir_quantize(input_data_str, disable_per_channel, fully_quantize, inference_type, input_data_type, output_data_type, enable_numeric_verify, enable_whole_model_verify, denylisted_ops, denylisted_nodes, enable_variable_quantization):
    """Wraps experimental mlir quantize model."""
    return _pywrap_toco_api.ExperimentalMlirQuantizeModel(input_data_str, disable_per_channel, fully_quantize, inference_type, input_data_type, output_data_type, enable_numeric_verify, enable_whole_model_verify, denylisted_ops, denylisted_nodes, enable_variable_quantization)
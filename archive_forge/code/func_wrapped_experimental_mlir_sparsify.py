from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import _pywrap_toco_api
def wrapped_experimental_mlir_sparsify(input_data_str):
    """Wraps experimental mlir sparsify model."""
    return _pywrap_toco_api.ExperimentalMlirSparsifyModel(input_data_str)
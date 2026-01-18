from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import _pywrap_toco_api
def wrapped_flat_buffer_file_to_mlir(model, input_is_filepath):
    """Wraps FlatBufferFileToMlir with lazy loader."""
    return _pywrap_toco_api.FlatBufferToMlir(model, input_is_filepath)
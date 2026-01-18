from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import _pywrap_toco_api
def wrapped_retrieve_collected_errors():
    """Wraps RetrieveCollectedErrors with lazy loader."""
    return _pywrap_toco_api.RetrieveCollectedErrors()
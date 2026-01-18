from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import _pywrap_toco_api
def wrapped_register_custom_opdefs(custom_opdefs_list):
    """Wraps RegisterCustomOpdefs with lazy loader."""
    return _pywrap_toco_api.RegisterCustomOpdefs(custom_opdefs_list)
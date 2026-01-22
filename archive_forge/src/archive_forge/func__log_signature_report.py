import collections
import os
import time
from tensorflow.python.keras.saving.utils_v1 import export_output as export_output_lib
from tensorflow.python.keras.saving.utils_v1 import mode_keys
from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.keras.saving.utils_v1.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
def _log_signature_report(signature_def_map, excluded_signatures):
    """Log a report of which signatures were produced."""
    sig_names_by_method_name = collections.defaultdict(list)
    for method_name in _FRIENDLY_METHOD_NAMES:
        sig_names_by_method_name[method_name] = []
    for signature_name, sig in signature_def_map.items():
        sig_names_by_method_name[sig.method_name].append(signature_name)
    for method_name, sig_names in sig_names_by_method_name.items():
        if method_name in _FRIENDLY_METHOD_NAMES:
            method_name = _FRIENDLY_METHOD_NAMES[method_name]
        logging.info('Signatures INCLUDED in export for {}: {}'.format(method_name, sig_names if sig_names else 'None'))
    if excluded_signatures:
        logging.info('Signatures EXCLUDED from export because they cannot be be served via TensorFlow Serving APIs:')
        for signature_name, message in excluded_signatures.items():
            logging.info("'{}' : {}".format(signature_name, message))
    if not signature_def_map:
        logging.warning('Export includes no signatures!')
    elif signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY not in signature_def_map:
        logging.warning('Export includes no default signature!')
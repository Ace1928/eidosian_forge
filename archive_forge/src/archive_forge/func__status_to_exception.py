from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
def _status_to_exception(status):
    try:
        error_class = errors.exception_type_from_error_code(status.code)
        e = error_class(None, None, status.message, status.payloads)
        logging.error_log('%s: %s' % (e.__class__.__name__, e))
        return e
    except KeyError:
        e = errors.UnknownError(None, None, status.message, status.code, status.payloads)
        logging.error_log('%s: %s' % (e.__class__.__name__, e))
        return e
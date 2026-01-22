from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
class _NotOkStatusException(Exception):
    """Exception class to handle not ok Status."""

    def __init__(self, message, code, payloads):
        super(_NotOkStatusException, self).__init__()
        self.message = message
        self.code = code
        self.payloads = payloads

    def __str__(self):
        e = _status_to_exception(self)
        return '%s: %s' % (e.__class__.__name__, e)
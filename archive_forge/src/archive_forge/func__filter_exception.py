import contextlib
import sys
import threading
import time
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _filter_exception(self, ex):
    """Check if the exception indicated in 'ex' should be ignored.

    This method examines `ex` to check if it is an exception that should be
    reported to the users.  If yes, it returns `ex` as is, otherwise it returns
    None.

    The code returns None for exception types listed in
    `_clean_stop_exception_types`.

    Args:
      ex: None, an `Exception`, or a Python `exc_info` tuple as returned by
        `sys.exc_info()`.

    Returns:
      ex or None.
    """
    if isinstance(ex, tuple):
        ex2 = ex[1]
    else:
        ex2 = ex
    if isinstance(ex2, self._clean_stop_exception_types):
        ex = None
    return ex
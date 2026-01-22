import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
def _log_deprecation(msg, *args, **kwargs):
    """Raises errors for deprecated methods if in strict mode, warns otherwise."""
    if strict_mode.STRICT_MODE:
        logging.error(msg, *args, **kwargs)
        raise RuntimeError('This behavior has been deprecated, which raises an error in strict mode.')
    else:
        logging.warning(msg, *args, **kwargs)
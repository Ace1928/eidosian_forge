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
def _call_location(outer=False):
    """Returns call location given level up from current call."""
    f = inspect.currentframe().f_back.f_back
    parent = f and f.f_back
    if outer and parent is not None:
        f = parent
    return '{}:{}'.format(f.f_code.co_filename, f.f_lineno)
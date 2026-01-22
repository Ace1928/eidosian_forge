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
def _get_arg_names_to_ok_vals():
    """Returns a dict mapping arg_name to DeprecatedArgSpec w/o position."""
    d = {}
    for name_or_tuple in deprecated_arg_names_or_tuples:
        if isinstance(name_or_tuple, tuple):
            d[name_or_tuple[0]] = DeprecatedArgSpec(-1, True, name_or_tuple[1])
        else:
            d[name_or_tuple] = DeprecatedArgSpec(-1, False, None)
    return d
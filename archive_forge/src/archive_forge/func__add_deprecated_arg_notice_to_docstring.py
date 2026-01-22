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
def _add_deprecated_arg_notice_to_docstring(doc, date, instructions, deprecated_names):
    """Adds a deprecation notice to a docstring for deprecated arguments."""
    deprecation_string = ', '.join(sorted(deprecated_names))
    return decorator_utils.add_notice_to_docstring(doc, instructions, 'DEPRECATED FUNCTION ARGUMENTS', '(deprecated arguments)', ['SOME ARGUMENTS ARE DEPRECATED: `(%s)`. They will be removed %s.' % (deprecation_string, 'in a future version' if date is None else 'after %s' % date), 'Instructions for updating:'], notice_type='Deprecated')
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
def _add_deprecated_arg_value_notice_to_docstring(doc, date, instructions, deprecated_name_value_dict):
    """Adds a deprecation notice to a docstring for deprecated arguments."""
    deprecation_string = ', '.join(('%s=%r' % (key, value) for key, value in sorted(deprecated_name_value_dict.items())))
    when = 'in a future version' if date is None else 'after %s' % date
    return decorator_utils.add_notice_to_docstring(doc, instructions, 'DEPRECATED FUNCTION ARGUMENT VALUES', '(deprecated argument values)', ['SOME ARGUMENT VALUES ARE DEPRECATED: `(%s)`. They will be removed %s.' % (deprecation_string, when), 'Instructions for updating:'], notice_type='Deprecated')